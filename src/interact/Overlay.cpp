#include "Overlay.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "ofxDrawSuite.h"
#include "ofxsInteract.h"

#include "../EditBlock.h"
#include "../ParamNames.h"

namespace shp {
namespace {

struct Colour {
  float r, g, b, a;
};

// Restrained on purpose: the overlay sits on top of the user's picture, so it
// has to stay legible without shouting over the image. Force 1 and Force 2 get
// distinct hues because the whole point is telling them apart at a glance.
constexpr Colour kForce1 = {0.20f, 0.72f, 1.00f, 1.00f};
constexpr Colour kForce2 = {1.00f, 0.62f, 0.18f, 1.00f};
constexpr Colour kRing = {1.00f, 1.00f, 1.00f, 0.34f};
constexpr Colour kShadow = {0.00f, 0.00f, 0.00f, 0.70f};
constexpr Colour kLabel = {0.92f, 0.94f, 0.98f, 1.00f};

// Fetched once, optionally: a host without the draw suite gets no overlay
// rather than a plugin that fails to load.
OfxDrawSuiteV1* drawSuite() {
  static OfxDrawSuiteV1* suite = const_cast<OfxDrawSuiteV1*>(
      static_cast<const OfxDrawSuiteV1*>(OFX::fetchSuite(kOfxDrawSuite, 1, /*optional*/ true)));
  return suite;
}

// The draw context is only valid for the duration of one draw action, and the
// Support library's DrawArgs does not carry it. The shim at the bottom of this
// file stashes it here immediately before delegating, and clears it straight
// after.
OfxDrawContextHandle gDrawContext = nullptr;

// Everything one event needs, snapshotted so nothing below re-reads a parameter
// halfway through and draws something self-inconsistent.
struct Ctx {
  OfxDrawSuiteV1* draw = nullptr;
  OfxDrawContextHandle ctx = nullptr;
  OfxPointD pixelScale{1.0, 1.0};
  OfxRectD rod{0.0, 0.0, 0.0, 0.0};

  // One screen pixel, in canonical units. Anything that should stay a fixed size
  // on screen has to be scaled by this; getting it wrong is the classic overlay
  // bug where handles become unusably tiny as soon as you zoom out.
  double sx(double px) const { return px * pixelScale.x; }
  double sy(double px) const { return px * pixelScale.y; }

  double width() const { return rod.x2 - rod.x1; }
  double height() const { return rod.y2 - rod.y1; }

  // Force positions are normalized with Y measured DOWNWARD from the top,
  // because that is the space src/core works in. Canonical image coordinates run
  // upward from the bottom. This is the single place the two meet; if handles
  // ever appear mirrored vertically, it is this function and nothing else.
  OfxPointD toCanonical(double nx, double ny) const {
    return OfxPointD{rod.x1 + nx * width(), rod.y2 - ny * height()};
  }
  void toNormalized(const OfxPointD& p, double& nx, double& ny) const {
    nx = width() > 0.0 ? (p.x - rod.x1) / width() : 0.0;
    ny = height() > 0.0 ? (rod.y2 - p.y) / height() : 0.0;
  }
};

void setColour(const Ctx& c, const Colour& col) {
  const OfxRGBAColourF v = {col.r, col.g, col.b, col.a};
  c.draw->setColour(c.ctx, &v);
}

void line(const Ctx& c, double x1, double y1, double x2, double y2) {
  const OfxPointD p[2] = {{x1, y1}, {x2, y2}};
  c.draw->draw(c.ctx, kOfxDrawPrimitiveLines, p, 2);
}

void fillRect(const Ctx& c, double x1, double y1, double x2, double y2) {
  const OfxPointD p[2] = {{x1, y1}, {x2, y2}};
  c.draw->draw(c.ctx, kOfxDrawPrimitiveRectangle, p, 2);
}

void ellipse(const Ctx& c, double cx, double cy, double rx, double ry) {
  const OfxPointD p[2] = {{cx - rx, cy - ry}, {cx + rx, cy + ry}};
  c.draw->draw(c.ctx, kOfxDrawPrimitiveEllipse, p, 2);
}

void text(const Ctx& c, const std::string& s, double x, double y) {
  const OfxPointD p = {x, y};
  c.draw->drawText(c.ctx, s.c_str(), &p, kOfxDrawTextAlignmentLeft);
}

// A square handle of fixed on-screen size, outlined so it stays visible over
// both bright and dark footage.
void handle(const Ctx& c, double x, double y, const Colour& fill, double screenSize = 5.0) {
  const double hx = c.sx(screenSize), hy = c.sy(screenSize);
  setColour(c, kShadow);
  fillRect(c, x - hx - c.sx(1.5), y - hy - c.sy(1.5), x + hx + c.sx(1.5), y + hy + c.sy(1.5));
  setColour(c, fill);
  fillRect(c, x - hx, y - hy, x + hx, y + hy);
}

// Hit test with a fixed on-screen tolerance, so grabbing a handle feels the same
// at any zoom level.
bool nearPoint(const Ctx& c, const OfxPointD& p, double x, double y, double screenRadius = 10.0) {
  return p.x >= x - c.sx(screenRadius) && p.x <= x + c.sx(screenRadius) &&
         p.y >= y - c.sy(screenRadius) && p.y <= y + c.sy(screenRadius);
}

std::string trimNum(double v, int decimals) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
  return buf;
}

// What a force looks like at one instant.
struct ForceView {
  bool enabled = false;
  double nx = 0.5, ny = 0.5;
  double radius = 0.4;   // fraction of frame width
  double strength = 1.0;
};

enum class Grab { None, Centre, Radius };

}  // namespace

// ---------------------------------------------------------------------------

namespace {

class ForceInteract : public OFX::OverlayInteract {
 public:
  ForceInteract(OfxInteractHandle handle, OFX::ImageEffect* effect)
      : OFX::OverlayInteract(handle), _effect(effect) {
    if (!_effect) return;
    // Slaving makes the host redraw the overlay when a value changes, so edits
    // typed into the Inspector move the handles immediately rather than on the
    // next viewer refresh.
    try {
      addParamToSlaveTo(_effect->fetchBooleanParam(kParamForce1Enabled));
      addParamToSlaveTo(_effect->fetchDouble2DParam(kParamForce1Pos));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce1Radius));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce1Strength));
      addParamToSlaveTo(_effect->fetchBooleanParam(kParamForce2Enabled));
      addParamToSlaveTo(_effect->fetchDouble2DParam(kParamForce2Pos));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce2Radius));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce2Strength));
      addParamToSlaveTo(_effect->fetchChoiceParam(kParamForceMode));
    } catch (...) {
    }
  }

  bool draw(const OFX::DrawArgs& args) override {
    Ctx c;
    if (!build(c, args.pixelScale, gDrawContext)) return false;

    // Force 2 first, so Force 1 lands on top -- it is the one enabled by
    // default and the one most drags are aimed at.
    drawForce(c, readForce(args.time, 1), kForce2, "Force 2");
    drawForce(c, readForce(args.time, 0), kForce1, "Force 1");
    return true;
  }

  bool penDown(const OFX::PenArgs& args) override {
    Ctx c;
    if (!build(c, args.pixelScale, nullptr)) return false;

    // Force 1 is tested first so that two overlapping forces resolve the same
    // way they draw: whatever is on top is what you grab.
    for (int i = 0; i < 2; ++i) {
      const ForceView f = readForce(args.time, i);
      if (!f.enabled) continue;

      const OfxPointD centre = c.toCanonical(f.nx, f.ny);
      if (nearPoint(c, args.penPosition, centre.x, centre.y)) {
        beginDrag(i, Grab::Centre);
        return true;
      }
      const double r = f.radius * c.width();
      if (nearPoint(c, args.penPosition, centre.x + r, centre.y)) {
        beginDrag(i, Grab::Radius);
        return true;
      }
    }
    return false;
  }

  bool penMotion(const OFX::PenArgs& args) override {
    if (_grab == Grab::None) return false;

    Ctx c;
    if (!build(c, args.pixelScale, nullptr)) return false;

    try {
      if (_grab == Grab::Centre) {
        double nx = 0.0, ny = 0.0;
        c.toNormalized(args.penPosition, nx, ny);
        posParam(_which)->setValue(nx, ny);
      } else {
        const ForceView f = readForce(args.time, _which);
        const OfxPointD centre = c.toCanonical(f.nx, f.ny);
        const double r = std::fabs(args.penPosition.x - centre.x);
        // Radius is a fraction of frame WIDTH, matching how Forces.h reads it.
        // Clamped away from zero because a zero radius divides in the falloff.
        const double frac = c.width() > 0.0 ? r / c.width() : 0.0;
        radiusParam(_which)->setValue(std::max(0.002, std::min(10.0, frac)));
      }
    } catch (...) {
    }
    return true;
  }

  bool penUp(const OFX::PenArgs&) override {
    if (_grab == Grab::None) return false;
    endDrag();
    return true;
  }

  // A drag that never gets its mouse-up -- released outside the viewer, a modal
  // dialog taking the release -- would otherwise leave the edit block open for
  // the rest of the session.
  void loseFocus(const OFX::FocusArgs&) override {
    if (_grab != Grab::None) endDrag();
  }

 private:
  OFX::Double2DParam* posParam(int i) const {
    return _effect->fetchDouble2DParam(i == 0 ? kParamForce1Pos : kParamForce2Pos);
  }
  OFX::DoubleParam* radiusParam(int i) const {
    return _effect->fetchDoubleParam(i == 0 ? kParamForce1Radius : kParamForce2Radius);
  }

  void beginDrag(int which, Grab what) {
    _which = which;
    _grab = what;
    // Bracketing groups the whole drag into one undo step instead of one per
    // mouse-move, and without it Fusion keeps serving cached frames after the
    // drag because it was never told an edit happened.
    BeginEdit(_effect, what == Grab::Centre ? "Move Force" : "Resize Force");
  }

  void endDrag() {
    _grab = Grab::None;
    EndEdit(_effect);
  }

  ForceView readForce(double t, int i) const {
    ForceView f;
    if (!_effect) return f;
    try {
      _effect->fetchBooleanParam(i == 0 ? kParamForce1Enabled : kParamForce2Enabled)
          ->getValueAtTime(t, f.enabled);
      posParam(i)->getValueAtTime(t, f.nx, f.ny);
      radiusParam(i)->getValueAtTime(t, f.radius);
      _effect->fetchDoubleParam(i == 0 ? kParamForce1Strength : kParamForce2Strength)
          ->getValueAtTime(t, f.strength);
    } catch (...) {
    }
    return f;
  }

  void drawForce(const Ctx& c, const ForceView& f, const Colour& col, const char* name) {
    // A disabled force is not drawn at all. Showing it greyed out sounds
    // helpful, but it puts a handle on screen that does nothing when dragged,
    // which is worse than it not being there.
    if (!f.enabled) return;

    const OfxPointD p = c.toCanonical(f.nx, f.ny);
    const double r = f.radius * c.width();

    // Reach of the force. Everything outside this circle is untouched by it, so
    // it is the single most useful thing the overlay can show.
    c.draw->setLineWidth(c.ctx, 1.0f);
    setColour(c, kRing);
    ellipse(c, p.x, p.y, r, r);

    setColour(c, col);
    ellipse(c, p.x, p.y, r * 0.5, r * 0.5);

    // Crosshair, sized in screen pixels so it stays findable at any zoom.
    const double armX = c.sx(14.0), armY = c.sy(14.0);
    setColour(c, kShadow);
    c.draw->setLineWidth(c.ctx, 3.0f);
    line(c, p.x - armX, p.y, p.x + armX, p.y);
    line(c, p.x, p.y - armY, p.x, p.y + armY);
    setColour(c, col);
    c.draw->setLineWidth(c.ctx, 1.0f);
    line(c, p.x - armX, p.y, p.x + armX, p.y);
    line(c, p.x, p.y - armY, p.x, p.y + armY);

    handle(c, p.x, p.y, col);
    handle(c, p.x + r, p.y, col, 4.0);  // drag this to change the radius

    // A negative strength pulls inward, which is invisible from the geometry
    // alone -- so it goes in the label rather than being left to surprise.
    const std::string sign = f.strength < 0.0 ? "  pull " : "  push ";
    setColour(c, kShadow);
    text(c, std::string(name) + sign + trimNum(std::fabs(f.strength), 2),
         p.x + c.sx(13.0), p.y + c.sy(11.0));
    setColour(c, kLabel);
    text(c, std::string(name) + sign + trimNum(std::fabs(f.strength), 2),
         p.x + c.sx(12.0), p.y + c.sy(12.0));
  }

  // Snapshots the viewport for one event. Never throws: an exception escaping
  // into the host mid-draw is far worse than a frame without an overlay.
  bool build(Ctx& out, const OfxPointD& pixelScale, OfxDrawContextHandle ctx) {
    try {
      if (!_effect || !drawSuite()) return false;

      OFX::Clip* src = _effect->fetchClip(kOfxImageEffectSimpleSourceClipName);
      if (!src || !src->isConnected()) return false;

      out.draw = drawSuite();
      out.ctx = ctx;
      out.pixelScale = pixelScale;
      out.rod = src->getRegionOfDefinition(0.0);

      if (out.rod.x2 - out.rod.x1 <= 0.0 || out.rod.y2 - out.rod.y1 <= 0.0) return false;
      // A degenerate pixel scale collapses every screen-relative size to zero
      // and the overlay silently vanishes.
      if (out.pixelScale.x <= 0.0) out.pixelScale.x = 1.0;
      if (out.pixelScale.y <= 0.0) out.pixelScale.y = 1.0;
      return true;
    } catch (...) {
      return false;
    }
  }

  OFX::ImageEffect* _effect = nullptr;
  Grab _grab = Grab::None;
  int _which = 0;
};

// Wraps the Support library's interact entry so the draw context, which only
// ever appears in the draw action's inArgs, can be captured on the way past.
// Everything else -- instance lifetime, pen dispatch, param slaving -- is left
// to the Support library.
class ForceOverlayDescriptor
    : public OFX::DefaultEffectOverlayDescriptor<ForceOverlayDescriptor, ForceInteract> {
 public:
  OfxPluginEntryPoint* getMainEntry() override { return &entry; }

 private:
  static OfxStatus entry(const char* action, const void* handle, OfxPropertySetHandle inArgs,
                         OfxPropertySetHandle outArgs) {
    const bool isDraw = std::strcmp(action, kOfxInteractActionDraw) == 0;
    if (isDraw && inArgs) {
      OFX::PropertySet p(inArgs);
      gDrawContext = static_cast<OfxDrawContextHandle>(
          p.propGetPointer(kOfxInteractPropDrawContext, 0, /*throwOnFailure*/ false));
    }

    const OfxStatus st =
        ForceOverlayDescriptor::overlayInteractMainEntry(action, handle, inArgs, outArgs);
    // The handle is only valid inside the draw action; holding it past that
    // would hand the widget a dangling context on the next pen event.
    if (isDraw) gDrawContext = nullptr;
    return st;
  }
};

}  // namespace

void registerOverlay(OFX::ImageEffectDescriptor& desc) {
  desc.setOverlayInteractDescriptor(new ForceOverlayDescriptor());
}

}  // namespace shp
