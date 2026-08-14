#include "Overlay.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "ofxDrawSuite.h"
#include "ofxsInteract.h"

#include "../EditBlock.h"
#include "../ParamNames.h"
#include "../SliceList.h"

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
  bool directional = false;
  double nx = 0.5, ny = 0.5;
  double radius = 0.4;     // fraction of frame width, radial only
  double direction = 90.0; // degrees, directional only
  double width = 0.5;      // fraction of frame width, directional only
  double strength = 1.0;
  double delay = 0.0;
};

enum class Grab { None, Centre, Radius, Width, Direction, SegEnd, SegNew, SegRotate };

constexpr Colour kCut = {1.00f, 0.28f, 0.34f, 0.95f};
constexpr Colour kObstacle = {0.25f, 0.95f, 0.72f, 0.95f};
constexpr Colour kPending = {1.00f, 0.85f, 0.30f, 0.95f};
constexpr Colour kDelete = {1.00f, 0.42f, 0.42f, 1.00f};

/// The two drawing tools share their gesture, their storage format and their
/// handles; only the parameter they write and the colour they draw in differ.
/// One code path for both is what keeps them behaving identically.
struct SegmentList {
  const char* param;
  Colour colour;
  const char* caption;
};

// How far from the centre the rotate handle sits, in screen pixels. Fixed on
// screen rather than in image space so the gesture feels the same at any zoom.
constexpr double kDirHandleDist = 64.0;

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
      addParamToSlaveTo(_effect->fetchChoiceParam(kParamForce1Shape));
      addParamToSlaveTo(_effect->fetchDouble2DParam(kParamForce1Pos));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce1Radius));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce1Direction));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce1Width));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce1Strength));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce1Delay));
      addParamToSlaveTo(_effect->fetchBooleanParam(kParamForce2Enabled));
      addParamToSlaveTo(_effect->fetchChoiceParam(kParamForce2Shape));
      addParamToSlaveTo(_effect->fetchDouble2DParam(kParamForce2Pos));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce2Radius));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce2Direction));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce2Width));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce2Strength));
      addParamToSlaveTo(_effect->fetchDoubleParam(kParamForce2Delay));
      addParamToSlaveTo(_effect->fetchChoiceParam(kParamForceMode));
      // Without these the cuts would not repaint until something else forced a
      // redraw, so a cut just drawn would appear to have been ignored.
      addParamToSlaveTo(_effect->fetchStringParam(kParamSlices));
      addParamToSlaveTo(_effect->fetchBooleanParam(kParamSliceTool));
      addParamToSlaveTo(_effect->fetchStringParam(kParamColliders));
      addParamToSlaveTo(_effect->fetchBooleanParam(kParamColliderTool));
      addParamToSlaveTo(_effect->fetchChoiceParam(kParamColliderShape));
    } catch (...) {
    }
  }

  bool draw(const OFX::DrawArgs& args) override {
    Ctx c;
    if (!build(c, args.pixelScale, gDrawContext)) return false;

    // Obstacles under cuts: a cut changes what the pieces ARE, so it is the more
    // important of the two to be able to see clearly.
    drawList(c, args.time, /*colliders=*/true);
    drawList(c, args.time, /*colliders=*/false);

    // Force 2 first, so Force 1 lands on top -- it is the one enabled by
    // default and the one most drags are aimed at.
    drawForce(c, readForce(args.time, 1), kForce2, "Force 2");
    drawForce(c, readForce(args.time, 0), kForce1, "Force 1");
    return true;
  }

  bool penDown(const OFX::PenArgs& args) override {
    Ctx c;
    if (!build(c, args.pixelScale, nullptr)) return false;

    // Endpoints of whichever tool is armed come first: once a line is drawn,
    // adjusting it is the commoner intent than drawing another through it. Only
    // the armed tool's endpoints are grabbable, so cuts and obstacles lying on
    // top of each other never fight over a click.
    for (int pass = 0; pass < 2; ++pass) {
      const bool colliders = pass == 0;
      if (!toolOn(args.time, colliders)) continue;

      std::vector<Entry> segs = readList(args.time, colliders);

      // Delete first. It is the smallest target and sits just outside the
      // shape, so anything else winning the hit test would make it unclickable
      // on a small obstacle.
      for (std::size_t i = 0; i < segs.size(); ++i) {
        const Handles h = handlesFor(c, segs[i]);
        if (!nearPoint(c, args.penPosition, h.remove.x, h.remove.y, 8.0)) continue;

        EditBlock block(_effect, colliders ? "Delete Obstacle" : "Delete Cut");
        segs.erase(segs.begin() + std::ptrdiff_t(i));
        writeList(segs, colliders);
        return true;   // a click, not a drag: nothing to keep hold of
      }

      for (std::size_t i = 0; i < segs.size(); ++i) {
        const Handles h = handlesFor(c, segs[i]);

        if (h.hasRotate && nearPoint(c, args.penPosition, h.rotate.x, h.rotate.y)) {
          _segIndex = int(i);
          _onColliders = colliders;
          _grab = Grab::SegRotate;
          BeginEdit(_effect, "Rotate Obstacle");
          return true;
        }

        const OfxPointD ends[2] = {h.p1, h.p2};
        for (int e = 0; e < 2; ++e) {
          if (nearPoint(c, args.penPosition, ends[e].x, ends[e].y)) {
            _segIndex = int(i);
            _segEnd = e;
            _onColliders = colliders;
            _grab = Grab::SegEnd;
            BeginEdit(_effect, colliders ? "Move Obstacle" : "Move Cut");
            return true;
          }
        }
      }
    }

    // With a tool armed, a drag anywhere else starts a new line.
    for (int pass = 0; pass < 2; ++pass) {
      const bool colliders = pass == 0;
      if (!toolOn(args.time, colliders)) continue;

      double nx = 0.0, ny = 0.0;
      c.toNormalized(args.penPosition, nx, ny);
      _pending = Entry{};
      // A cut is always a line; an obstacle is whatever the Draw menu says. The
      // shape is captured now, at the start of the drag, so changing the menu
      // mid-drag cannot turn a half-drawn box into a circle.
      _pending.shape = colliders ? pendingShape(args.time) : ColliderShape::Line;
      _pending.x1 = _pending.x2 = float(nx);
      _pending.y1 = _pending.y2 = float(ny);
      _onColliders = colliders;
      _grab = Grab::SegNew;
      BeginEdit(_effect, colliders ? "Add Obstacle" : "Add Cut");
      return true;
    }

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

      if (f.directional) {
        double dx, dy;
        dirVector(f.direction, dx, dy);
        const double aLen = c.sx(kDirHandleDist);
        if (nearPoint(c, args.penPosition, centre.x + dx * aLen, centre.y + dy * aLen)) {
          beginDrag(i, Grab::Direction);
          return true;
        }
        const double half = f.width * c.width() * 0.5;
        if (nearPoint(c, args.penPosition, centre.x - dy * half, centre.y + dx * half)) {
          beginDrag(i, Grab::Width);
          return true;
        }
      } else {
        const double r = f.radius * c.width();
        if (nearPoint(c, args.penPosition, centre.x + r, centre.y)) {
          beginDrag(i, Grab::Radius);
          return true;
        }
      }
    }
    return false;
  }

  bool penMotion(const OFX::PenArgs& args) override {
    if (_grab == Grab::None) return false;

    Ctx c;
    if (!build(c, args.pixelScale, nullptr)) return false;

    // Cuts are handled before the forces because they have their own storage --
    // a string parameter rather than a pair of doubles.
    if (_grab == Grab::SegNew) {
      double nx = 0.0, ny = 0.0;
      c.toNormalized(args.penPosition, nx, ny);
      _pending.x2 = float(nx);
      _pending.y2 = float(ny);
      return true;
    }
    if (_grab == Grab::SegRotate) {
      try {
        std::vector<Entry> segs = readList(args.time, _onColliders);
        if (_segIndex >= 0 && _segIndex < int(segs.size())) {
          Entry& e = segs[std::size_t(_segIndex)];
          const OfxPointD mid = c.toCanonical(0.5 * (double(e.x1) + double(e.x2)),
                                              0.5 * (double(e.y1) + double(e.y2)));
          // Angle from the box's centre to the pointer, measured so that zero
          // points along the box's own "up". Negated on the way out for the
          // same y-axis reason boxCorners documents.
          const double dx = args.penPosition.x - mid.x;
          const double dy = args.penPosition.y - mid.y;
          double deg = -std::atan2(dx, dy) * 180.0 / 3.14159265358979323846;
          if (deg < 0.0) deg += 360.0;
          e.angle = float(deg);
          writeList(segs, _onColliders);
        }
      } catch (...) {
      }
      return true;
    }
    if (_grab == Grab::SegEnd) {
      try {
        std::vector<Entry> segs = readList(args.time, _onColliders);
        if (_segIndex >= 0 && _segIndex < int(segs.size())) {
          Entry& e = segs[std::size_t(_segIndex)];
          double nx = 0.0, ny = 0.0;
          c.toNormalized(args.penPosition, nx, ny);

          // A rotated box stores its corners UNROTATED, so the pointer has to be
          // turned back into that frame before it can be written down. Without
          // this, dragging a corner of a turned box moves it along the screen
          // axes and the box appears to shear.
          if (e.shape == ColliderShape::Rect && e.angle != 0.0f) {
            const double cxN = 0.5 * (double(e.x1) + double(e.x2));
            const double cyN = 0.5 * (double(e.y1) + double(e.y2));
            const OfxPointD mid = c.toCanonical(cxN, cyN);
            const double a = double(e.angle) * 3.14159265358979323846 / 180.0;
            const double ca = std::cos(a), sa = std::sin(a);
            const double rx = args.penPosition.x - mid.x;
            const double ry = args.penPosition.y - mid.y;
            const OfxPointD un{mid.x + rx * ca - ry * sa, mid.y + rx * sa + ry * ca};
            c.toNormalized(un, nx, ny);
          }

          if (_segEnd == 0) {
            e.x1 = float(nx);
            e.y1 = float(ny);
          } else {
            e.x2 = float(nx);
            e.y2 = float(ny);
          }
          writeList(segs, _onColliders);
        }
      } catch (...) {
      }
      return true;
    }

    try {
      const ForceView f = readForce(args.time, _which);
      const OfxPointD centre = c.toCanonical(f.nx, f.ny);
      const double relX = args.penPosition.x - centre.x;
      const double relY = args.penPosition.y - centre.y;

      switch (_grab) {
        case Grab::Centre: {
          double nx = 0.0, ny = 0.0;
          c.toNormalized(args.penPosition, nx, ny);
          posParam(_which)->setValue(nx, ny);
          break;
        }
        case Grab::Radius: {
          // Radius is a fraction of frame WIDTH, matching how Forces.h reads
          // it. Clamped away from zero because a zero radius divides in the
          // falloff.
          const double frac = c.width() > 0.0 ? std::fabs(relX) / c.width() : 0.0;
          radiusParam(_which)->setValue(std::max(0.002, std::min(10.0, frac)));
          break;
        }
        case Grab::Width: {
          // Only the component across the band counts, so sliding along the
          // direction does not fight the drag.
          double dx, dy;
          dirVector(f.direction, dx, dy);
          const double across = std::fabs(relX * -dy + relY * dx);
          const double frac = c.width() > 0.0 ? (across * 2.0) / c.width() : 0.0;
          widthParam(_which)->setValue(std::max(0.004, std::min(10.0, frac)));
          break;
        }
        case Grab::Direction: {
          // atan2 in the canonical frame, inverted back into the core's
          // convention where 180 is down. Same single negation as dirVector.
          const double deg = std::atan2(relX, relY) * 180.0 / 3.14159265358979323846;
          directionParam(_which)->setValue(deg < 0.0 ? deg + 360.0 : deg);
          break;
        }
        case Grab::None:
          break;
      }
    } catch (...) {
    }
    return true;
  }

  bool penUp(const OFX::PenArgs& args) override {
    if (_grab == Grab::None) return false;

    if (_grab == Grab::SegNew) {
      // A drag that went nowhere is a click, not a line. Committing it would
      // leave a zero-length entry that does nothing and still has to be found
      // and deleted.
      Ctx c;
      const bool haveCtx = build(c, args.pixelScale, nullptr);
      const double dx = double(_pending.x2 - _pending.x1) * (haveCtx ? c.width() : 1.0);
      const double dy = double(_pending.y2 - _pending.y1) * (haveCtx ? c.height() : 1.0);
      if (dx * dx + dy * dy >= 9.0) {
        try {
          std::vector<Entry> segs = readList(args.time, _onColliders);
          if (int(segs.size()) < (_onColliders ? kMaxColliders : kMaxSlices)) {
            segs.push_back(_pending);
            writeList(segs, _onColliders);
          }
        } catch (...) {
        }
      }
    }

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
  OFX::DoubleParam* widthParam(int i) const {
    return _effect->fetchDoubleParam(i == 0 ? kParamForce1Width : kParamForce2Width);
  }
  OFX::DoubleParam* directionParam(int i) const {
    return _effect->fetchDoubleParam(i == 0 ? kParamForce1Direction : kParamForce2Direction);
  }

  void beginDrag(int which, Grab what) {
    _which = which;
    _grab = what;
    // Bracketing groups the whole drag into one undo step instead of one per
    // mouse-move, and without it Fusion keeps serving cached frames after the
    // drag because it was never told an edit happened.
    const char* label = what == Grab::Centre      ? "Move Force"
                        : what == Grab::Direction ? "Aim Force"
                                                  : "Resize Force";
    BeginEdit(_effect, label);
  }

  void endDrag() {
    _grab = Grab::None;
    EndEdit(_effect);
  }

  static SegmentList listFor(bool colliders) {
    return colliders ? SegmentList{kParamColliders, kObstacle, "Collider Tool: drag to draw"}
                     : SegmentList{kParamSlices, kCut, "Slice Tool: drag to cut"};
  }

  bool toolOn(double t, bool colliders) const {
    bool on = false;
    try {
      _effect->fetchBooleanParam(colliders ? kParamColliderTool : kParamSliceTool)
          ->getValueAtTime(t, on);
    } catch (...) {
    }
    return on;
  }

  // Both lists are read into the same shape here -- two control points plus a
  // shape tag -- which is what lets one set of drag handles edit all of them.
  // For a cut the tag is always Line and is ignored.
  struct Entry {
    ColliderShape shape = ColliderShape::Line;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float angle = 0;   // Box only
  };

  /// Every handle an entry offers, computed once and used by BOTH the drawing
  /// and the hit test. Two functions deriving these independently is how a
  /// handle ends up drawn somewhere it cannot be grabbed.
  struct Handles {
    OfxPointD p1{0, 0};
    OfxPointD p2{0, 0};
    OfxPointD rotate{0, 0};
    OfxPointD remove{0, 0};
    bool hasRotate = false;
    OfxPointD corner[4];   // Box only: the rotated outline
  };

  /// The box's four corners in canonical space.
  ///
  /// The angle is stored in the core's top-down frame where positive turns
  /// clockwise on screen. Canonical coordinates run y-UP, so reproducing the
  /// same turn here means rotating by the NEGATED angle. This is the only place
  /// that conversion happens.
  static void boxCorners(const Ctx& c, const Entry& e, OfxPointD* out) {
    const double cxN = 0.5 * (double(e.x1) + double(e.x2));
    const double cyN = 0.5 * (double(e.y1) + double(e.y2));
    const OfxPointD mid = c.toCanonical(cxN, cyN);
    const double hx = std::fabs(double(e.x2) - double(e.x1)) * 0.5 * c.width();
    const double hy = std::fabs(double(e.y2) - double(e.y1)) * 0.5 * c.height();

    const double a = -double(e.angle) * 3.14159265358979323846 / 180.0;
    const double ca = std::cos(a), sa = std::sin(a);
    const double lx[4] = {-hx, hx, hx, -hx};
    const double ly[4] = {-hy, -hy, hy, hy};
    for (int i = 0; i < 4; ++i) {
      out[i].x = mid.x + lx[i] * ca - ly[i] * sa;
      out[i].y = mid.y + lx[i] * sa + ly[i] * ca;
    }
  }

  Handles handlesFor(const Ctx& c, const Entry& e) const {
    Handles h;
    h.p1 = c.toCanonical(e.x1, e.y1);
    h.p2 = c.toCanonical(e.x2, e.y2);

    // Bounds of what is actually drawn, so the delete handle sits clear of the
    // shape whatever the shape is.
    double minX, minY, maxX, maxY;

    if (e.shape == ColliderShape::Rect) {
      boxCorners(c, e, h.corner);
      minX = maxX = h.corner[0].x;
      minY = maxY = h.corner[0].y;
      for (int i = 1; i < 4; ++i) {
        minX = std::min(minX, h.corner[i].x);
        maxX = std::max(maxX, h.corner[i].x);
        minY = std::min(minY, h.corner[i].y);
        maxY = std::max(maxY, h.corner[i].y);
      }
      // The rotate handle rides on the box's own "up", pushed clear of the edge
      // so it never lands under a corner handle.
      const double mx = 0.5 * (h.corner[0].x + h.corner[2].x);
      const double my = 0.5 * (h.corner[0].y + h.corner[2].y);
      const double ux = h.corner[0].x - h.corner[3].x;
      const double uy = h.corner[0].y - h.corner[3].y;
      const double ulen = std::sqrt(ux * ux + uy * uy);
      const double nx = ulen > 1e-6 ? ux / ulen : 0.0;
      const double ny = ulen > 1e-6 ? uy / ulen : 1.0;
      const double edge = 0.5 * ulen;
      h.rotate.x = mx + nx * (edge + c.sx(26.0));
      h.rotate.y = my + ny * (edge + c.sy(26.0));
      h.hasRotate = true;
    } else if (e.shape == ColliderShape::Circle) {
      const double r = std::sqrt((h.p2.x - h.p1.x) * (h.p2.x - h.p1.x) +
                                 (h.p2.y - h.p1.y) * (h.p2.y - h.p1.y));
      minX = h.p1.x - r;
      maxX = h.p1.x + r;
      minY = h.p1.y - r;
      maxY = h.p1.y + r;
    } else {
      minX = std::min(h.p1.x, h.p2.x);
      maxX = std::max(h.p1.x, h.p2.x);
      minY = std::min(h.p1.y, h.p2.y);
      maxY = std::max(h.p1.y, h.p2.y);
    }

    // Top-right of the shape's bounds, offset outward. Fixed in screen pixels,
    // so it stays reachable at any zoom and never collides with a control point.
    h.remove.x = maxX + c.sx(13.0);
    h.remove.y = maxY + c.sy(13.0);
    (void)minX;
    (void)minY;
    return h;
  }

  std::vector<Entry> readList(double t, bool colliders) const {
    std::string s;
    try {
      _effect->fetchStringParam(listFor(colliders).param)->getValueAtTime(t, s);
    } catch (...) {
    }

    std::vector<Entry> out;
    if (colliders) {
      for (const Collider& c : ParseColliders(s))
        out.push_back(Entry{c.shape, c.x1, c.y1, c.x2, c.y2, c.angle});
    } else {
      for (const Slice& c : ParseSegments<Slice>(s, kMaxSlices))
        out.push_back(Entry{ColliderShape::Line, c.x1, c.y1, c.x2, c.y2, 0.0f});
    }
    return out;
  }

  void writeList(const std::vector<Entry>& segs, bool colliders) const {
    std::string text;
    if (colliders) {
      std::vector<Collider> cs;
      for (const Entry& e : segs) {
        Collider c;
        c.shape = e.shape;
        c.x1 = e.x1;
        c.y1 = e.y1;
        c.x2 = e.x2;
        c.y2 = e.y2;
        c.angle = e.angle;
        cs.push_back(c);
      }
      text = FormatColliders(cs);
    } else {
      std::vector<Slice> ss;
      for (const Entry& e : segs) {
        Slice c;
        c.x1 = e.x1;
        c.y1 = e.y1;
        c.x2 = e.x2;
        c.y2 = e.y2;
        ss.push_back(c);
      }
      text = FormatSlices(ss);
    }
    _effect->fetchStringParam(listFor(colliders).param)->setValue(text);
  }

  ColliderShape pendingShape(double t) const {
    int v = 0;
    try {
      _effect->fetchChoiceParam(kParamColliderShape)->getValueAtTime(t, v);
    } catch (...) {
    }
    return ColliderShape(v < 0 ? 0 : (v > 2 ? 2 : v));
  }

  // One entry drawn, whatever it is.
  void drawEntry(const Ctx& c, const Entry& e, const Colour& col, bool handles) {
    const Handles h = handlesFor(c, e);

    // Dark pass then light, so a line stays readable over both bright and dark
    // footage -- an obstacle you cannot see is one you cannot move.
    for (int pass = 0; pass < 2; ++pass) {
      setColour(c, pass == 0 ? kShadow : col);
      c.draw->setLineWidth(c.ctx, pass == 0 ? 3.0f : 1.0f);

      switch (e.shape) {
        case ColliderShape::Rect:
          c.draw->draw(c.ctx, kOfxDrawPrimitiveLineLoop, h.corner, 4);
          break;
        case ColliderShape::Circle: {
          const double r = std::sqrt((h.p2.x - h.p1.x) * (h.p2.x - h.p1.x) +
                                     (h.p2.y - h.p1.y) * (h.p2.y - h.p1.y));
          ellipse(c, h.p1.x, h.p1.y, r, r);
          break;
        }
        case ColliderShape::Line:
        default:
          line(c, h.p1.x, h.p1.y, h.p2.x, h.p2.y);
          break;
      }
    }

    if (!handles) return;
    handle(c, h.p1.x, h.p1.y, col, 4.0);
    handle(c, h.p2.x, h.p2.y, col, 4.0);

    if (h.hasRotate) {
      // Tethered to the box, so it reads as belonging to it rather than as a
      // stray point floating nearby.
      setColour(c, kShadow);
      c.draw->setLineWidth(c.ctx, 3.0f);
      const double mx = 0.5 * (h.corner[0].x + h.corner[2].x);
      const double my = 0.5 * (h.corner[0].y + h.corner[2].y);
      line(c, mx, my, h.rotate.x, h.rotate.y);
      setColour(c, col);
      c.draw->setLineWidth(c.ctx, 1.0f);
      line(c, mx, my, h.rotate.x, h.rotate.y);
      handle(c, h.rotate.x, h.rotate.y, col, 4.0);
    }

    // Delete: a cross rather than a filled square, so it never reads as one more
    // thing to drag.
    const double r = c.sx(5.0), ry = c.sy(5.0);
    setColour(c, kShadow);
    c.draw->setLineWidth(c.ctx, 4.0f);
    line(c, h.remove.x - r, h.remove.y - ry, h.remove.x + r, h.remove.y + ry);
    line(c, h.remove.x - r, h.remove.y + ry, h.remove.x + r, h.remove.y - ry);
    setColour(c, kDelete);
    c.draw->setLineWidth(c.ctx, 2.0f);
    line(c, h.remove.x - r, h.remove.y - ry, h.remove.x + r, h.remove.y + ry);
    line(c, h.remove.x - r, h.remove.y + ry, h.remove.x + r, h.remove.y - ry);
  }

  void drawList(const Ctx& c, double t, bool colliders) {
    const std::vector<Entry> segs = readList(t, colliders);
    const bool tool = toolOn(t, colliders);
    const SegmentList def = listFor(colliders);

    // Handles only while the matching tool is armed. Left on permanently they
    // would clutter the viewer for anyone who has finished drawing and is now
    // tuning the physics.
    for (const Entry& e : segs) drawEntry(c, e, def.colour, tool);

    // Whatever is currently being dragged, in a different colour so it is
    // obviously not committed yet.
    if (_grab == Grab::SegNew && _onColliders == colliders) {
      drawEntry(c, _pending, kPending, false);
    }

    if (tool) {
      setColour(c, kLabel);
      text(c, def.caption, c.rod.x1 + c.sx(16.0), c.rod.y2 - c.sy(20.0));
    }
  }

  ForceView readForce(double t, int i) const {
    ForceView f;
    if (!_effect) return f;
    try {
      _effect->fetchBooleanParam(i == 0 ? kParamForce1Enabled : kParamForce2Enabled)
          ->getValueAtTime(t, f.enabled);
      int shape = 0;
      _effect->fetchChoiceParam(i == 0 ? kParamForce1Shape : kParamForce2Shape)
          ->getValueAtTime(t, shape);
      f.directional = shape == kForceDirectional;
      posParam(i)->getValueAtTime(t, f.nx, f.ny);
      radiusParam(i)->getValueAtTime(t, f.radius);
      directionParam(i)->getValueAtTime(t, f.direction);
      widthParam(i)->getValueAtTime(t, f.width);
      _effect->fetchDoubleParam(i == 0 ? kParamForce1Strength : kParamForce2Strength)
          ->getValueAtTime(t, f.strength);
      _effect->fetchDoubleParam(i == 0 ? kParamForce1Delay : kParamForce2Delay)
          ->getValueAtTime(t, f.delay);
    } catch (...) {
    }
    return f;
  }

  // Canonical-space unit vector for a force direction.
  //
  // The core works top-down and canonical coordinates run bottom-up, so the y
  // component is negated exactly once, here. 180 degrees must point at the
  // bottom of the viewer, the same way gravity does.
  static void dirVector(double degrees, double& dx, double& dy) {
    const double a = degrees * 3.14159265358979323846 / 180.0;
    dx = std::sin(a);
    dy = std::cos(a);  // canonical y is up, so "down" (180) gives cos = -1
  }

  void drawForce(const Ctx& c, const ForceView& f, const Colour& col, const char* name) {
    // A disabled force is not drawn at all. Showing it greyed out sounds
    // helpful, but it puts a handle on screen that does nothing when dragged,
    // which is worse than it not being there.
    if (!f.enabled) return;

    const OfxPointD p = c.toCanonical(f.nx, f.ny);
    c.draw->setLineWidth(c.ctx, 1.0f);

    if (f.directional) {
      double dx, dy;
      dirVector(f.direction, dx, dy);
      const double px = -dy, py = dx;  // perpendicular, for the band edges
      const double half = f.width * c.width() * 0.5;

      // The band has no end along the direction, so the edges are drawn right
      // across the frame and beyond -- a line that stopped would suggest a
      // reach the force does not actually have.
      const double reach = std::sqrt(c.width() * c.width() + c.height() * c.height());

      setColour(c, kRing);
      for (int s = -1; s <= 1; s += 2) {
        const double ox = px * half * s, oy = py * half * s;
        line(c, p.x + ox - dx * reach, p.y + oy - dy * reach, p.x + ox + dx * reach,
             p.y + oy + dy * reach);
      }
      // Centre line, where the force is strongest.
      setColour(c, col);
      line(c, p.x - dx * reach, p.y - dy * reach, p.x + dx * reach, p.y + dy * reach);

      // Arrow along the direction, so which way it blows is unambiguous.
      const double aLen = c.sx(kDirHandleDist);
      const double tipX = p.x + dx * aLen, tipY = p.y + dy * aLen;
      const double headX = c.sx(10.0), headY = c.sy(10.0);
      setColour(c, kShadow);
      c.draw->setLineWidth(c.ctx, 3.0f);
      line(c, p.x, p.y, tipX, tipY);
      setColour(c, col);
      c.draw->setLineWidth(c.ctx, 1.0f);
      line(c, p.x, p.y, tipX, tipY);
      line(c, tipX, tipY, tipX - dx * headX + px * headX * 0.6,
           tipY - dy * headY + py * headY * 0.6);
      line(c, tipX, tipY, tipX - dx * headX - px * headX * 0.6,
           tipY - dy * headY - py * headY * 0.6);

      handle(c, p.x, p.y, col);
      handle(c, tipX, tipY, col, 4.0);                              // rotate
      handle(c, p.x + px * half, p.y + py * half, col, 4.0);        // width
    } else {
      const double r = f.radius * c.width();

      // Reach of the force. Everything outside this circle is untouched by it,
      // so it is the single most useful thing the overlay can show.
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
    }

    // A negative strength pulls inward and a delay means nothing happens yet --
    // both invisible from the geometry alone, so they go in the label rather
    // than being left to surprise.
    std::string label = std::string(name) + (f.strength < 0.0 ? "  pull " : "  push ") +
                        trimNum(std::fabs(f.strength), 2);
    if (f.delay > 0.0) label += "  +" + trimNum(f.delay, 0) + "f";

    setColour(c, kShadow);
    text(c, label, p.x + c.sx(13.0), p.y + c.sy(11.0));
    setColour(c, kLabel);
    text(c, label, p.x + c.sx(12.0), p.y + c.sy(12.0));
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

  Entry _pending;          ///< what is being dragged, not yet committed
  int _segIndex = -1;
  int _segEnd = 0;
  bool _onColliders = false;  ///< which list the current drag belongs to
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
