// OFX host glue. Everything interesting lives under src/core; this file only
// fetches images, reads parameters, and maps OFX coordinates into the flat
// top-down space the core code works in.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"

#include "ClipTime.h"
#include "EditBlock.h"
#include "ParamNames.h"
#include "ParamRegistry.h"
#include "SliceList.h"
#include "core/Ballistic.h"
#include "core/Cells.h"
#include "core/Compositor.h"
#include "core/ImageView.h"
#include "core/Segmentation.h"
#include "core/SimCache.h"
#include "interact/Overlay.h"

#if SHP_WITH_CUDA
#include "render/CudaRender.h"
#endif

#define kPluginName "Shatter Physics"
#define kPluginGrouping "Stylize"
#define kPluginIdentifier "com.johnlwi.ShatterPhysics"
#define kPluginDescription                                                        \
  "Breaks the image into pieces and throws them, with forces, gravity and "       \
  "optional rigid-body collisions. Pieces come from a generated pattern "         \
  "(glass, bricks, grid), from a custom map, or straight from the alpha "         \
  "channel -- one connected region per piece, so a logo or a title shatters "     \
  "into its own parts with no roto."

namespace {

// Parameter kind, deduced from the descriptor type so the registry cannot
// disagree with what was actually defined.
inline int kindOf(OFX::DoubleParamDescriptor*) { return shp::kPKDouble; }
inline int kindOf(OFX::Double2DParamDescriptor*) { return shp::kPKDouble2D; }
inline int kindOf(OFX::IntParamDescriptor*) { return shp::kPKInt; }
inline int kindOf(OFX::BooleanParamDescriptor*) { return shp::kPKBool; }
inline int kindOf(OFX::ChoiceParamDescriptor*) { return shp::kPKChoice; }
inline int kindOf(OFX::RGBAParamDescriptor*) { return shp::kPKRGBA; }
inline int kindOf(OFX::StringParamDescriptor*) { return shp::kPKString; }
// Anything else -- groups, pages, push buttons -- holds nothing worth storing.
inline int kindOf(...) { return shp::kPKNone; }

// Single funnel for adding a value-bearing parameter to a page.
//
// Every parameter declares that it invalidates the whole cache, for the reason
// text-animator and multi-transform both measured: Resolve keeps a per-frame
// verdict about an effect and will happily keep serving frames it rendered
// earlier. The OFX default, eCacheInvalidateValueChange, means "invalidate only
// the frames this parameter's own keyframe covers". That is right for a
// keyframed effect and wrong here -- nothing in this plugin is keyframed, the
// parameters are constants the simulation interprets against the render time,
// so every parameter affects every frame. Without this, edits made while the
// playhead sits outside the shatter's range appear to do nothing at all, and
// only a manual cache purge fixes it.
template <class T>
void addParam(OFX::PageParamDescriptor* page, T* p, OFX::GroupParamDescriptor* group) {
  p->setCacheInvalidation(OFX::eCacheInvalidateValueAll);
  shp::registerParam(p->getName(), kindOf(p));
  if (group) p->setParent(*group);
  page->addChild(*p);
}

// For parameters that change nothing about the rendered image. Purging every
// cached frame because someone toggled a viewer control would be a stall for no
// reason.
template <class T>
void addParamUI(OFX::PageParamDescriptor* page, T* p, OFX::GroupParamDescriptor* group) {
  shp::registerParam(p->getName(), kindOf(p));
  if (group) p->setParent(*group);
  page->addChild(*p);
}

// ChoiceParam is the one param type without a return-value getter, and a host
// that has persisted an index we no longer define would otherwise walk off the
// end of a switch.
int choiceAt(OFX::ChoiceParam* p, double t, int lo, int hi) {
  int v = 0;
  p->getValueAtTime(t, v);
  return std::max(lo, std::min(hi, v));
}

// OFX images are bottom-up: y=0 is the bottom row. The core code is top-down
// (row 0 is the top, +y goes downward). Converting here, at the single boundary,
// is what keeps "gravity points down" meaning the same thing in Resolve as in
// the offline harness -- and it is why the harness can hold a golden image at
// all. The flip is a negative row stride, so no pixels are moved.
shp::ImageView makeView(OFX::Image* img, const OfxRectI& region) {
  shp::ImageView v;
  if (!img) return v;
  void* base = img->getPixelAddress(region.x1, region.y2 - 1);  // topmost row
  if (!base) return v;
  v.data = static_cast<float*>(base);
  v.width = region.x2 - region.x1;
  v.height = region.y2 - region.y1;
  v.rowStride = -std::ptrdiff_t(img->getRowBytes()) / std::ptrdiff_t(sizeof(float));
  return v;
}

// Re-shattering every frame would be wasteful while scrubbing a static title, so
// keep the last result and reuse it when the input and the shape settings match.
//
// Held by shared_ptr so a cache hit costs a refcount bump. Copying a
// ShatterGeometry per frame would mean memcpy'ing the label image -- 8 MB at
// 1080p, 33 MB at UHD -- on every single render, while holding the mutex.
struct AnalysisCache {
  std::mutex mutex;
  uint64_t hash = 0;
  uint64_t mapHash = 0;
  int width = 0, height = 0;
  shp::ShapeParams params;
  std::shared_ptr<const shp::ShatterGeometry> geom;
};

class ShatterPhysicsPlugin : public OFX::ImageEffect {
 public:
  ~ShatterPhysicsPlugin() override {
#if SHP_WITH_CUDA
    if (_cuda) shp::CudaDestroy(_cuda);
#endif
  }

  explicit ShatterPhysicsPlugin(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    if (getContext() == OFX::eContextGeneral) _mapClip = fetchClip(shp::kClipShatterMap);

    _pattern = fetchChoiceParam(shp::kParamPattern);
    _cellCount = fetchIntParam(shp::kParamCellCount);
    _cellJitter = fetchDoubleParam(shp::kParamCellJitter);
    _brickRows = fetchIntParam(shp::kParamBrickRows);
    _brickOffset = fetchDoubleParam(shp::kParamBrickOffset);
    _mapLevels = fetchIntParam(shp::kParamMapLevels);
    _alphaThreshold = fetchDoubleParam(shp::kParamAlphaThreshold);
    _minPieceArea = fetchDoubleParam(shp::kParamMinPieceArea);
    _bridgeRadius = fetchDoubleParam(shp::kParamBridgeRadius);
    _shapeSeed = fetchIntParam(shp::kParamShapeSeed);
    _slices = fetchStringParam(shp::kParamSlices);
    _sliceTool = fetchBooleanParam(shp::kParamSliceTool);

    _force1Enabled = fetchBooleanParam(shp::kParamForce1Enabled);
    _force1Shape = fetchChoiceParam(shp::kParamForce1Shape);
    _force1Pos = fetchDouble2DParam(shp::kParamForce1Pos);
    _force1Radius = fetchDoubleParam(shp::kParamForce1Radius);
    _force1Direction = fetchDoubleParam(shp::kParamForce1Direction);
    _force1Width = fetchDoubleParam(shp::kParamForce1Width);
    _force1Strength = fetchDoubleParam(shp::kParamForce1Strength);
    _force1Delay = fetchDoubleParam(shp::kParamForce1Delay);
    _force2Enabled = fetchBooleanParam(shp::kParamForce2Enabled);
    _force2Shape = fetchChoiceParam(shp::kParamForce2Shape);
    _force2Pos = fetchDouble2DParam(shp::kParamForce2Pos);
    _force2Radius = fetchDoubleParam(shp::kParamForce2Radius);
    _force2Direction = fetchDoubleParam(shp::kParamForce2Direction);
    _force2Width = fetchDoubleParam(shp::kParamForce2Width);
    _force2Strength = fetchDoubleParam(shp::kParamForce2Strength);
    _force2Delay = fetchDoubleParam(shp::kParamForce2Delay);
    _forceMode = fetchChoiceParam(shp::kParamForceMode);
    _waveSpeed = fetchDoubleParam(shp::kParamWaveSpeed);
    _falloff = fetchChoiceParam(shp::kParamFalloff);

    _simMode = fetchChoiceParam(shp::kParamSimMode);
    _rotationSpeed = fetchDoubleParam(shp::kParamRotationSpeed);
    _randomness = fetchDoubleParam(shp::kParamRandomness);
    _viscosity = fetchDoubleParam(shp::kParamViscosity);
    _massVariance = fetchDoubleParam(shp::kParamMassVariance);
    _gravity = fetchDoubleParam(shp::kParamGravity);
    _gravityDirection = fetchDoubleParam(shp::kParamGravityDirection);
    _bounciness = fetchDoubleParam(shp::kParamBounciness);
    _friction = fetchDoubleParam(shp::kParamFriction);
    _collideShards = fetchBooleanParam(shp::kParamCollideShards);
    _floorEnabled = fetchBooleanParam(shp::kParamFloorEnabled);
    _floorY = fetchDoubleParam(shp::kParamFloorY);
    _wallsEnabled = fetchBooleanParam(shp::kParamWallsEnabled);
    _substeps = fetchIntParam(shp::kParamSubsteps);

    _startFrame = fetchIntParam(shp::kParamStartFrame);
    _seed = fetchIntParam(shp::kParamSeed);

    _fadeStart = fetchDoubleParam(shp::kParamFadeStart);
    _fadeDuration = fetchDoubleParam(shp::kParamFadeDuration);
    _shrinkOnFade = fetchBooleanParam(shp::kParamShrinkOnFade);
    _edgeAntialias = fetchBooleanParam(shp::kParamEdgeAntialias);
    _showPieces = fetchBooleanParam(shp::kParamShowPieces);
    _motionBlur = fetchBooleanParam(shp::kParamMotionBlur);
    _shutterAngle = fetchDoubleParam(shp::kParamShutterAngle);
    _blurSamples = fetchIntParam(shp::kParamBlurSamples);

    syncVisibility();
  }

  void render(const OFX::RenderArguments& args) override;
  void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) override;

  // Always false, never conditionally.
  //
  // Answering "yes, identity" lets the host substitute the source image and skip
  // this effect for that frame -- and it caches that verdict. Once the shatter
  // does start moving, frames the host already decided were pass-through stay
  // pass-through. Returning false unconditionally costs a copy on frames before
  // the shatter begins, and is the difference between an effect that works when
  // scrubbed and one that works only when played forward from the top.
  bool isIdentity(const OFX::IsIdentityArguments&, OFX::Clip*&, double&) override {
    return false;
  }

 private:
  shp::ShapeParams readShape(double t) const;
  shp::ForceParams readForces(double t) const;
  shp::PhysicsParams readPhysics(double t) const;
  shp::RenderParams readRender(double t) const;
  void syncVisibility();

  std::shared_ptr<const shp::ShatterGeometry> geometryFor(const shp::ConstImageView& src,
                                                          const shp::ConstImageView* map,
                                                          const shp::ShapeParams& sp);

  OFX::Clip* _dstClip = nullptr;
  OFX::Clip* _srcClip = nullptr;
  OFX::Clip* _mapClip = nullptr;

  OFX::ChoiceParam* _pattern = nullptr;
  OFX::IntParam* _cellCount = nullptr;
  OFX::DoubleParam* _cellJitter = nullptr;
  OFX::IntParam* _brickRows = nullptr;
  OFX::DoubleParam* _brickOffset = nullptr;
  OFX::IntParam* _mapLevels = nullptr;
  OFX::DoubleParam* _alphaThreshold = nullptr;
  OFX::DoubleParam* _minPieceArea = nullptr;
  OFX::DoubleParam* _bridgeRadius = nullptr;
  OFX::IntParam* _shapeSeed = nullptr;
  OFX::StringParam* _slices = nullptr;
  OFX::BooleanParam* _sliceTool = nullptr;

  OFX::BooleanParam* _force1Enabled = nullptr;
  OFX::ChoiceParam* _force1Shape = nullptr;
  OFX::Double2DParam* _force1Pos = nullptr;
  OFX::DoubleParam* _force1Radius = nullptr;
  OFX::DoubleParam* _force1Direction = nullptr;
  OFX::DoubleParam* _force1Width = nullptr;
  OFX::DoubleParam* _force1Strength = nullptr;
  OFX::DoubleParam* _force1Delay = nullptr;
  OFX::BooleanParam* _force2Enabled = nullptr;
  OFX::ChoiceParam* _force2Shape = nullptr;
  OFX::Double2DParam* _force2Pos = nullptr;
  OFX::DoubleParam* _force2Radius = nullptr;
  OFX::DoubleParam* _force2Direction = nullptr;
  OFX::DoubleParam* _force2Width = nullptr;
  OFX::DoubleParam* _force2Strength = nullptr;
  OFX::DoubleParam* _force2Delay = nullptr;
  OFX::ChoiceParam* _forceMode = nullptr;
  OFX::DoubleParam* _waveSpeed = nullptr;
  OFX::ChoiceParam* _falloff = nullptr;

  OFX::ChoiceParam* _simMode = nullptr;
  OFX::DoubleParam* _rotationSpeed = nullptr;
  OFX::DoubleParam* _randomness = nullptr;
  OFX::DoubleParam* _viscosity = nullptr;
  OFX::DoubleParam* _massVariance = nullptr;
  OFX::DoubleParam* _gravity = nullptr;
  OFX::DoubleParam* _gravityDirection = nullptr;
  OFX::DoubleParam* _bounciness = nullptr;
  OFX::DoubleParam* _friction = nullptr;
  OFX::BooleanParam* _collideShards = nullptr;
  OFX::BooleanParam* _floorEnabled = nullptr;
  OFX::DoubleParam* _floorY = nullptr;
  OFX::BooleanParam* _wallsEnabled = nullptr;
  OFX::IntParam* _substeps = nullptr;

  OFX::IntParam* _startFrame = nullptr;
  OFX::IntParam* _seed = nullptr;

  OFX::DoubleParam* _fadeStart = nullptr;
  OFX::DoubleParam* _fadeDuration = nullptr;
  OFX::BooleanParam* _shrinkOnFade = nullptr;
  OFX::BooleanParam* _edgeAntialias = nullptr;
  OFX::BooleanParam* _showPieces = nullptr;
  OFX::BooleanParam* _motionBlur = nullptr;
  OFX::DoubleParam* _shutterAngle = nullptr;
  OFX::IntParam* _blurSamples = nullptr;

  mutable AnalysisCache _cache;

  // The rigid-body timeline. Carries its own lock, and is only touched when the
  // simulation mode calls for it.
  shp::SimCache _sim;

  // Scratch reused across frames so a 1000-shard render is not 1000 allocations.
  // Guarded by the same lock as the cache: the effect is eRenderInstanceSafe, so
  // the host may call render on two threads for one instance.
  std::mutex _scratchMutex;
  std::vector<shp::ShardPose> _poses;
  std::vector<shp::ShardDraw> _draws;
  shp::TileGrid _tiles;

#if SHP_WITH_CUDA
  bool renderCuda(const OFX::RenderArguments& args, OFX::Image* src, OFX::Image* dst,
                  OFX::Image* mapImg);

  shp::CudaContext* _cuda = nullptr;
  /// Host copy of the frame, downloaded only when the device-side hash says the
  /// content changed. Segmentation is a serial CPU job, so it needs real pixels.
  std::vector<float> _hostFrame;
  /// Identity of the geometry whose labels are currently resident on the device.
  const shp::ShatterGeometry* _devLabelsFor = nullptr;
#endif
};

// ------------------------------------------------------------- parameter reads

shp::ShapeParams ShatterPhysicsPlugin::readShape(double t) const {
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern(choiceAt(_pattern, t, 0, 5));
  _cellCount->getValueAtTime(t, sp.cellCount);
  double d = 0.0;
  _cellJitter->getValueAtTime(t, d);
  sp.cellJitter = float(d);
  _brickRows->getValueAtTime(t, sp.brickRows);
  _brickOffset->getValueAtTime(t, d);
  sp.brickOffset = float(d);
  _mapLevels->getValueAtTime(t, sp.mapLevels);
  _alphaThreshold->getValueAtTime(t, d);
  sp.alphaThreshold = float(d);
  _minPieceArea->getValueAtTime(t, d);
  // The control is a percentage of the frame, because 0.0002 is not a number
  // anyone can reason about while dragging a slider.
  sp.minPieceArea = float(d * 0.01);
  _bridgeRadius->getValueAtTime(t, d);
  sp.bridgeRadius = float(d * 0.01);
  _shapeSeed->getValueAtTime(t, sp.shapeSeed);

  // Hand-drawn cuts apply on top of every pattern, not just Manual Slices, so
  // a generated shatter can be corrected rather than replaced.
  {
    std::string s;
    _slices->getValueAtTime(t, s);
    sp.slices = shp::ParseSlices(s);
  }

  // Collision hulls cost real time on a thousand-shard frame, and the ballistic
  // mode never looks at them.
  sp.buildPolygons = choiceAt(_simMode, t, 0, 1) == shp::kSimRigidBody;
  return sp;
}

shp::ForceParams ShatterPhysicsPlugin::readForces(double t) const {
  shp::ForceParams fp;
  double x = 0.0, y = 0.0, d = 0.0;

  auto readField = [&](shp::ForceField& f, OFX::BooleanParam* en, OFX::ChoiceParam* shape,
                       OFX::Double2DParam* pos, OFX::DoubleParam* radius,
                       OFX::DoubleParam* dir, OFX::DoubleParam* width,
                       OFX::DoubleParam* strength, OFX::DoubleParam* delay) {
    en->getValueAtTime(t, f.enabled);
    f.shape = shp::ForceShape(choiceAt(shape, t, 0, 1));
    pos->getValueAtTime(t, x, y);
    f.posX = float(x);
    f.posY = float(y);
    radius->getValueAtTime(t, d);
    f.radius = float(d);
    dir->getValueAtTime(t, d);
    f.direction = float(d);
    width->getValueAtTime(t, d);
    f.width = float(d);
    strength->getValueAtTime(t, d);
    f.strength = float(d);
    delay->getValueAtTime(t, d);
    f.delay = float(d);
  };

  readField(fp.f1, _force1Enabled, _force1Shape, _force1Pos, _force1Radius, _force1Direction,
            _force1Width, _force1Strength, _force1Delay);
  readField(fp.f2, _force2Enabled, _force2Shape, _force2Pos, _force2Radius, _force2Direction,
            _force2Width, _force2Strength, _force2Delay);

  fp.mode = shp::ForceMode(choiceAt(_forceMode, t, 0, 2));
  _waveSpeed->getValueAtTime(t, d);
  fp.waveSpeed = float(d);
  fp.falloff = shp::Falloff(choiceAt(_falloff, t, 0, 2));

  _startFrame->getValueAtTime(t, fp.startFrame);
  _seed->getValueAtTime(t, fp.seed);
  return fp;
}

shp::PhysicsParams ShatterPhysicsPlugin::readPhysics(double t) const {
  shp::PhysicsParams pp;
  double d = 0.0;
  _rotationSpeed->getValueAtTime(t, d);
  pp.rotationSpeed = float(d);
  _randomness->getValueAtTime(t, d);
  pp.randomness = float(d);
  _viscosity->getValueAtTime(t, d);
  pp.viscosity = float(d);
  _massVariance->getValueAtTime(t, d);
  pp.massVariance = float(d);
  _gravity->getValueAtTime(t, d);
  pp.gravity = float(d);
  _gravityDirection->getValueAtTime(t, d);
  pp.gravityDirection = float(d);
  _bounciness->getValueAtTime(t, d);
  pp.bounciness = float(d);
  _friction->getValueAtTime(t, d);
  pp.friction = float(d);
  _collideShards->getValueAtTime(t, pp.collideShards);
  _floorEnabled->getValueAtTime(t, pp.floorEnabled);
  _floorY->getValueAtTime(t, d);
  pp.floorY = float(d);
  _wallsEnabled->getValueAtTime(t, pp.wallsEnabled);
  _substeps->getValueAtTime(t, pp.substeps);
  return pp;
}

shp::RenderParams ShatterPhysicsPlugin::readRender(double t) const {
  shp::RenderParams rp;
  double d = 0.0;
  _fadeStart->getValueAtTime(t, d);
  rp.fadeStart = float(d);
  _fadeDuration->getValueAtTime(t, d);
  rp.fadeDuration = float(d);
  _shrinkOnFade->getValueAtTime(t, rp.shrinkOnFade);
  return rp;
}

// ------------------------------------------------------------------ visibility

// One place, called from the constructor and from changedParam, and never from
// render. Splitting visibility across per-parameter handlers is how a control
// ends up visible in one state and hidden in an equivalent one, depending on
// which order the user happened to click things in.
void ShatterPhysicsPlugin::syncVisibility() {
  const int pattern = choiceAt(_pattern, 0.0, 0, 5);
  const int mode = choiceAt(_forceMode, 0.0, 0, 2);
  const int sim = choiceAt(_simMode, 0.0, 0, 1);

  const bool voronoi = pattern == shp::kPatternVoronoi;
  const bool bricks = pattern == shp::kPatternBricks || pattern == shp::kPatternGrid;
  const bool map = pattern == shp::kPatternMap;

  _cellCount->setIsSecret(!voronoi);
  _cellJitter->setIsSecret(!voronoi);
  _brickRows->setIsSecret(!bricks);
  _brickOffset->setIsSecret(pattern != shp::kPatternBricks);
  _mapLevels->setIsSecret(!map);

  _waveSpeed->setIsSecret(mode != shp::kForceRadialWave);

  // Radius belongs to a radial force, direction and width to a directional one.
  // Showing all three at once invites setting a width that does nothing.
  const bool dir1 = choiceAt(_force1Shape, 0.0, 0, 1) == shp::kForceDirectional;
  _force1Radius->setIsSecret(dir1);
  _force1Direction->setIsSecret(!dir1);
  _force1Width->setIsSecret(!dir1);

  const bool dir2 = choiceAt(_force2Shape, 0.0, 0, 1) == shp::kForceDirectional;
  _force2Radius->setIsSecret(dir2);
  _force2Direction->setIsSecret(!dir2);
  _force2Width->setIsSecret(!dir2);

  const bool rigid = sim == shp::kSimRigidBody;
  _bounciness->setIsSecret(!rigid);
  _friction->setIsSecret(!rigid);
  _collideShards->setIsSecret(!rigid);
  _floorEnabled->setIsSecret(!rigid);
  _floorY->setIsSecret(!rigid);
  _wallsEnabled->setIsSecret(!rigid);
  _substeps->setIsSecret(!rigid);

  // The fade sub-controls mean nothing while the fade is switched off.
  double dur = 0.0;
  _fadeDuration->getValue(dur);
  _fadeStart->setIsSecret(dur <= 0.0);
  _shrinkOnFade->setIsSecret(dur <= 0.0);

  bool mb = false;
  _motionBlur->getValue(mb);
  _shutterAngle->setIsSecret(!mb);
  _blurSamples->setIsSecret(!mb);
}

void ShatterPhysicsPlugin::changedParam(const OFX::InstanceChangedArgs&,
                                        const std::string& name) {
  // A throw out of changedParam takes the host's parameter dispatch with it, so
  // nothing is allowed to escape.
  try {
    if (name == shp::kParamClearSlices) {
      // Bracketed so it is one undo step, and so Fusion is told the value moved
      // -- without that it keeps serving frames cut with the old slices.
      shp::EditBlock block(this, "Clear Cuts");
      _slices->setValue("");
      return;
    }
    if (name == shp::kParamPattern || name == shp::kParamForceMode ||
        name == shp::kParamSimMode || name == shp::kParamFadeDuration ||
        name == shp::kParamMotionBlur || name == shp::kParamForce1Shape ||
        name == shp::kParamForce2Shape) {
      syncVisibility();
    }
  } catch (...) {
  }
}

// -------------------------------------------------------------------- geometry

std::shared_ptr<const shp::ShatterGeometry> ShatterPhysicsPlugin::geometryFor(
    const shp::ConstImageView& src, const shp::ConstImageView* map,
    const shp::ShapeParams& sp) {
  const uint64_t h = shp::hashAlpha(src, sp.alphaThreshold);
  const uint64_t mh = (map && map->valid()) ? shp::hashRGB(*map) : 0;

  std::lock_guard<std::mutex> lock(_cache.mutex);
  if (_cache.geom && _cache.hash == h && _cache.mapHash == mh && _cache.width == src.width &&
      _cache.height == src.height && _cache.params == sp) {
    return _cache.geom;  // a hit is a refcount bump, not a 33 MB copy
  }

  auto geom = shp::buildGeometry(src, map, sp);
  _cache.geom = geom;
  _cache.hash = h;
  _cache.mapHash = mh;
  _cache.width = src.width;
  _cache.height = src.height;
  _cache.params = sp;
  return geom;
}

#if SHP_WITH_CUDA
// The CUDA path.
//
// By the time this is called the host has handed over DEVICE pointers, so
// nothing here may dereference a pixel on the CPU. There is deliberately no
// fallback to the CPU loop on failure: the addresses are already device memory
// and the CPU cannot touch them, so a failure has to be a black frame rather
// than a crash.
bool ShatterPhysicsPlugin::renderCuda(const OFX::RenderArguments& args, OFX::Image* src,
                                      OFX::Image* dst, OFX::Image* mapImg) {
  const OfxRectI bounds = dst->getBounds();
  const int w = bounds.x2 - bounds.x1;
  const int h = bounds.y2 - bounds.y1;
  if (w <= 0 || h <= 0) return false;

  auto* srcBase = static_cast<const float*>(src->getPixelData());
  auto* dstBase = static_cast<float*>(dst->getPixelData());
  if (!srcBase || !dstBase) return false;

  const std::ptrdiff_t srcStride =
      std::ptrdiff_t(src->getRowBytes()) / std::ptrdiff_t(sizeof(float));
  const std::ptrdiff_t dstStride =
      std::ptrdiff_t(dst->getRowBytes()) / std::ptrdiff_t(sizeof(float));
  if (srcStride == 0 || dstStride == 0) return false;

  void* stream = args.pCudaStream;
  if (!_cuda) _cuda = shp::CudaCreate();
  if (!_cuda) return false;

  const double t = args.time;
  const shp::ShapeParams sp = readShape(t);

  // The fingerprint is computed ON THE DEVICE and read back as eight bytes, so
  // the 133 MB frame only crosses the bus when the content has actually
  // changed. Without this the CUDA path would download every frame to feed the
  // segmentation and be slower than the CPU path it replaced.
  const uint64_t hash =
      shp::CudaHashSource(_cuda, srcBase, w, h, srcStride, sp.alphaThreshold, stream);

  std::shared_ptr<const shp::ShatterGeometry> geom;
  {
    std::lock_guard<std::mutex> lock(_cache.mutex);
    if (_cache.geom && _cache.hash == hash && _cache.width == w && _cache.height == h &&
        _cache.params == sp) {
      geom = _cache.geom;
    }
  }

  if (!geom) {
    // Cache miss: bring the frame down once and label it on the CPU.
    _hostFrame.resize(std::size_t(w) * std::size_t(h) * 4);
    if (!shp::CudaDownload(_cuda, srcBase, w, h, srcStride, _hostFrame.data(), stream))
      return false;

    // The download is packed bottom-up exactly as the device holds it; this
    // view flips it to the top-down space src/core works in, the same way
    // makeView does for the CPU path.
    shp::ConstImageView hostView;
    hostView.data = _hostFrame.data() + std::size_t(h - 1) * std::size_t(w) * 4;
    hostView.width = w;
    hostView.height = h;
    hostView.rowStride = -std::ptrdiff_t(w) * 4;

    // A custom map would need its own download; not supported on the GPU path
    // yet, so it falls back to the generated pattern rather than reading device
    // memory on the host.
    (void)mapImg;
    geom = shp::buildGeometry(hostView, nullptr, sp);

    std::lock_guard<std::mutex> lock(_cache.mutex);
    _cache.geom = geom;
    _cache.hash = hash;
    _cache.mapHash = 0;
    _cache.width = w;
    _cache.height = h;
    _cache.params = sp;
  }

  if (!geom || geom->empty()) return false;

  const shp::ForceParams fp = readForces(t);
  const shp::PhysicsParams pp = readPhysics(t);
  const shp::RenderParams rp = readRender(t);

  shp::CompositeParams cp;
  _edgeAntialias->getValueAtTime(t, cp.edgeAntialias);
  _showPieces->getValueAtTime(t, cp.showPieces);

  shp::BlurParams blur;
  _motionBlur->getValueAtTime(t, blur.enabled);
  {
    double d = 0.0;
    _shutterAngle->getValueAtTime(t, d);
    blur.shutterAngle = float(d);
    _blurSamples->getValueAtTime(t, blur.samples);
  }

  const double clipT = shp::toClipTime(this, t);

  shp::FrameScale fs;
  fs.width = float(geom->width);
  fs.height = float(geom->height);
  fs.meanArea = geom->meanArea;
  fs.meanRadius = geom->meanRadius;

  const bool rigid = choiceAt(_simMode, t, 0, 1) == shp::kSimRigidBody;

  std::lock_guard<std::mutex> lock(_scratchMutex);
  if (rigid) {
    _sim.getPoses(geom, fp, pp, rp, fs, float(clipT), _poses);
  } else {
    shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, float(clipT), _poses);
  }
  shp::BuildDrawList(*geom, _poses, blur, _draws);
  shp::BuildTileGrid(_draws, geom->width, geom->height, _tiles);

  // The label image is 33 MB at 4K, so it is uploaded when the geometry object
  // changes rather than per frame. Pointer identity is the key, for the same
  // reason the other caches use it: the analysis cache guarantees one object per
  // content, and the shared_ptr above keeps the address from being recycled.
  if (_devLabelsFor != geom.get()) {
    if (!shp::CudaUploadLabels(_cuda, geom->labelImage.data(), geom->width, geom->height,
                               stream))
      return false;
    _devLabelsFor = geom.get();
  }

  // Top-down views over the host's bottom-up buffers: point at the last row and
  // negate the stride, exactly as makeView does on the CPU path, so the kernel
  // and the label image agree on which way is up.
  const float* srcTop = srcBase + (std::size_t(h - 1) * std::size_t(srcStride));
  float* dstTop = dstBase + (std::size_t(h - 1) * std::size_t(dstStride));

  return shp::CudaComposite(_cuda, srcTop, dstTop, w, h, -srcStride, -dstStride, _draws.data(),
                            int(_draws.size()), _tiles.start.data(), _tiles.tileCount(),
                            _tiles.shardIdx.data(), int(_tiles.shardIdx.size()), _tiles.tilesX,
                            _tiles.tilesY, cp, stream);
}
#endif  // SHP_WITH_CUDA

// ---------------------------------------------------------------------- render

void ShatterPhysicsPlugin::render(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip ? _dstClip->fetchImage(args.time) : nullptr);
  if (!dst) return;

  const OfxRectI region = dst->getBounds();

  std::unique_ptr<OFX::Image> src(
      _srcClip && _srcClip->isConnected() ? _srcClip->fetchImage(args.time) : nullptr);

#if SHP_WITH_CUDA
  // Taken BEFORE any view is built, because on this path the pixel pointers are
  // device memory and makeView's arithmetic is the only thing that may touch
  // them. There is no fallback to the CPU loop if this fails, for the same
  // reason.
  if (args.isEnabledCudaRender && shp::CudaAvailable() && src) {
    std::unique_ptr<OFX::Image> mapImg(
        (_mapClip && _mapClip->isConnected()) ? _mapClip->fetchImage(args.time) : nullptr);
    if (renderCuda(args, src.get(), dst.get(), mapImg.get())) return;
    // Failed: leave the frame black rather than reading device memory on the
    // host. A black frame is a visible bug; a host-side dereference of a device
    // pointer is a crash that takes Resolve with it.
    return;
  }
#endif

  shp::ImageView dstView = makeView(dst.get(), region);
  if (!dstView.valid()) return;

  auto clearOut = [&]() {
    for (int y = 0; y < dstView.height; ++y)
      std::memset(dstView.row(y), 0, std::size_t(dstView.width) * 4 * sizeof(float));
  };

  if (!src) {
    clearOut();
    return;
  }
  const shp::ImageView srcView = makeView(src.get(), region);
  if (!srcView.valid()) {
    clearOut();
    return;
  }

  // Never write parameters from here. Render runs on host worker threads outside
  // any edit block, and a host that dispatches the resulting changedParam back
  // onto one of them deadlocks against the cache mutex.
  const double t = args.time;
  const shp::ShapeParams sp = readShape(t);
  const shp::ForceParams fp = readForces(t);
  const shp::PhysicsParams pp = readPhysics(t);
  const shp::RenderParams rp = readRender(t);

  shp::CompositeParams cp;
  _edgeAntialias->getValueAtTime(t, cp.edgeAntialias);
  _showPieces->getValueAtTime(t, cp.showPieces);

  shp::BlurParams blur;
  _motionBlur->getValueAtTime(t, blur.enabled);
  {
    double d = 0.0;
    _shutterAngle->getValueAtTime(t, d);
    blur.shutterAngle = float(d);
    _blurSamples->getValueAtTime(t, blur.samples);
  }

  std::unique_ptr<OFX::Image> mapImg(
      (sp.pattern == shp::Pattern::Map && _mapClip && _mapClip->isConnected())
          ? _mapClip->fetchImage(args.time)
          : nullptr);
  shp::ImageView mapView;
  if (mapImg) mapView = makeView(mapImg.get(), mapImg->getBounds());
  const shp::ConstImageView mapConst(mapView);

  auto geom = geometryFor(srcView, mapConst.valid() ? &mapConst : nullptr, sp);
  if (!geom || geom->empty()) {
    clearOut();
    return;
  }

  // Clip-relative frames. Everything downstream is in frames, never seconds.
  const double clipT = shp::toClipTime(this, t);

  shp::FrameScale fs;
  fs.width = float(geom->width);
  fs.height = float(geom->height);
  fs.meanArea = geom->meanArea;
  fs.meanRadius = geom->meanRadius;

  const bool rigid = choiceAt(_simMode, t, 0, 1) == shp::kSimRigidBody;

  std::lock_guard<std::mutex> lock(_scratchMutex);
  if (rigid) {
    // Steps forward from the start frame on a cache miss, then serves stored
    // poses. Scrubbing backward is a lookup; the first render deep into a clip
    // pays for the whole prefix once.
    _sim.getPoses(geom, fp, pp, rp, fs, float(clipT), _poses);
  } else {
    shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, float(clipT), _poses);
  }
  shp::BuildDrawList(*geom, _poses, blur, _draws);
  shp::BuildTileGrid(_draws, geom->width, geom->height, _tiles);
  shp::Composite(dstView, srcView, *geom, _draws, _tiles, cp);
}

// ---------------------------------------------------------------- factory

mDeclarePluginFactory(ShatterPhysicsFactory, {}, {});

void ShatterPhysicsFactory::describe(OFX::ImageEffectDescriptor& desc) {
  desc.setLabels(kPluginName, kPluginName, kPluginName);
  desc.setPluginGrouping(kPluginGrouping);
  desc.setPluginDescription(kPluginDescription);

  desc.addSupportedContext(OFX::eContextFilter);
  desc.addSupportedContext(OFX::eContextGeneral);
  desc.addSupportedBitDepth(OFX::eBitDepthFloat);

  desc.setSingleInstance(false);
  desc.setHostFrameThreading(false);
  desc.setSupportsMultiResolution(false);

  // Mandatory, and must stay off. Both the alpha segmentation and the cell
  // pattern are whole-image measurements: handed a tile, the labeller would
  // shatter every piece that crosses the tile boundary into two, and each tile
  // would number its pieces from 1 independently.
  desc.setSupportsTiles(false);

  // The simulation reaches backward in its OWN cache, never into the source clip
  // at another time, so no temporal clip access is needed. Asking for it would
  // make the host fetch frames it does not otherwise need.
  desc.setTemporalClipAccess(false);

  desc.setRenderTwiceAlways(false);
  desc.setSupportsMultipleClipDepths(false);
  desc.setRenderThreadSafety(OFX::eRenderInstanceSafe);

#if SHP_WITH_CUDA
  // Gated on a device actually being present. Claiming CUDA on a machine
  // without one would have the host hand over device pointers that cannot be
  // serviced -- and by then there is no way back to the CPU loop, because the
  // pixel addresses are already device memory.
  //
  // Without this, Resolve pulls every frame down to system memory to run us and
  // pushes the result back. At 4K float RGBA that is ~133 MB each way, which is
  // more than the entire cost of compositing the frame and is what stops a CPU
  // plugin from playing back in realtime.
  if (shp::CudaAvailable()) {
    desc.setSupportsCudaRender(true);
    desc.setSupportsCudaStream(true);
  }
#endif

  // Plugin-level property, so it belongs in describe() rather than
  // describeInContext() -- the host caches the descriptor and would not
  // necessarily pick it up from the per-context pass.
  shp::registerOverlay(desc);
}

void ShatterPhysicsFactory::describeInContext(OFX::ImageEffectDescriptor& desc,
                                              OFX::ContextEnum context) {
  OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
  src->addSupportedComponent(OFX::ePixelComponentRGBA);
  src->setTemporalClipAccess(false);
  src->setSupportsTiles(false);
  src->setIsMask(false);

  if (context == OFX::eContextGeneral) {
    // Second input for the custom shatter map. Optional, so the effect still
    // works as a plain filter with nothing attached.
    OFX::ClipDescriptor* map = desc.defineClip(shp::kClipShatterMap);
    map->addSupportedComponent(OFX::ePixelComponentRGBA);
    map->setTemporalClipAccess(false);
    map->setSupportsTiles(false);
    map->setIsMask(false);
    map->setOptional(true);
  }

  OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
  dst->addSupportedComponent(OFX::ePixelComponentRGBA);
  dst->setSupportsTiles(false);

  // One page. Resolve reports maxPages = 0, so the structure has to come from
  // groups rather than from pages.
  OFX::PageParamDescriptor* page = desc.definePageParam("Controls");

  auto defineGroup = [&](const char* name, const char* label, bool open) {
    OFX::GroupParamDescriptor* g = desc.defineGroupParam(name);
    g->setLabels(label, label, label);
    g->setOpen(open);
    page->addChild(*g);
    return g;
  };

  // ------------------------------------------------------------------- Shape
  OFX::GroupParamDescriptor* gShape = defineGroup("grpShape", "Shape", true);
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(shp::kParamPattern);
    p->setLabels("Pattern", "Pattern", "Pattern");
    p->setHint(
        "How the image is divided. Glass makes irregular shards; Bricks and Grid "
        "make regular ones. Alpha Pieces uses no pattern at all -- every "
        "connected region of alpha becomes one piece, so a logo breaks into the "
        "parts it was drawn from and a title breaks into its own letters. Custom "
        "Map reads regions of flat colour from the second input.");
    p->appendOption("Glass (Voronoi)");
    p->appendOption("Bricks");
    p->appendOption("Grid");
    p->appendOption("Alpha Pieces");
    p->appendOption("Custom Map");
    p->appendOption("Manual Slices");
    p->setDefault(shp::kPatternVoronoi);
    addParam(page, p, gShape);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(shp::kParamSliceTool);
    p->setLabels("Slice Tool", "Slice Tool", "Slice Tool");
    p->setHint(
        "Turn this on and drag across the viewer to cut. Each drag adds one "
        "cut, which separates whatever it crosses all the way through -- a cut "
        "that stops halfway through a piece does not split it, exactly like a "
        "real knife. Drag the square handles at either end to adjust a cut "
        "afterwards.\n\n"
        "Cuts apply on top of whichever pattern is selected, so you can break "
        "the frame automatically and then fix the two places it got wrong. "
        "Choose the Manual Slices pattern to work from nothing but your own "
        "cuts.");
    p->setDefault(false);
    addParamUI(page, p, gShape);  // a viewer mode, not a value the render reads
  }
  {
    OFX::StringParamDescriptor* p = desc.defineStringParam(shp::kParamSlices);
    p->setLabels("Cuts", "Cuts", "Cuts");
    p->setHint(
        "The cuts you have drawn, as x1,y1,x2,y2 in fractions of the frame. "
        "Editable by hand, and safe to copy between clips.");
    p->setDefault("");
    addParam(page, p, gShape);
  }
  {
    OFX::PushButtonParamDescriptor* p = desc.definePushButtonParam(shp::kParamClearSlices);
    p->setLabels("Clear Cuts", "Clear Cuts", "Clear Cuts");
    p->setHint("Remove every hand-drawn cut. Undo brings them back.");
    addParamUI(page, p, gShape);
  }
  {
    OFX::IntParamDescriptor* p = desc.defineIntParam(shp::kParamCellCount);
    p->setLabels("Pieces", "Pieces", "Pieces");
    p->setHint(
        "Roughly how many shards to cut the frame into. The actual count comes "
        "out lower, because cells that miss the image entirely are discarded.");
    p->setRange(1, 5000);
    p->setDisplayRange(4, 600);
    p->setDefault(50);
    addParam(page, p, gShape);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamCellJitter);
    p->setLabels("Irregularity", "Irregularity", "Irregularity");
    p->setHint("0 gives an even lattice of shards; 1 scatters them like real broken glass.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.75);
    addParam(page, p, gShape);
  }
  {
    OFX::IntParamDescriptor* p = desc.defineIntParam(shp::kParamBrickRows);
    p->setLabels("Rows", "Rows", "Rows");
    p->setHint("How many courses the frame is divided into.");
    p->setRange(1, 500);
    p->setDisplayRange(1, 40);
    p->setDefault(8);
    addParam(page, p, gShape);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamBrickOffset);
    p->setLabels("Row Offset", "Offset", "Row Offset");
    p->setHint("How far alternate rows slide sideways. 0.5 is a normal running bond.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.5);
    addParam(page, p, gShape);
  }
  {
    OFX::IntParamDescriptor* p = desc.defineIntParam(shp::kParamMapLevels);
    p->setLabels("Map Levels", "Levels", "Map Levels");
    p->setHint(
        "How finely the map's colours are quantised before regions are found. "
        "Lower merges near-identical colours into one piece.");
    p->setRange(2, 64);
    p->setDisplayRange(2, 32);
    p->setDefault(16);
    addParam(page, p, gShape);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamAlphaThreshold);
    p->setLabels("Alpha Threshold", "Alpha", "Alpha Threshold");
    p->setHint(
        "How opaque a pixel must be to count as part of the image. Raise it if "
        "soft edges are producing slivers of debris.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.15);
    addParam(page, p, gShape);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamMinPieceArea);
    p->setLabels("Min Piece Size", "Min Size", "Min Piece Size");
    p->setHint(
        "Pieces smaller than this share of the frame are discarded as debris. "
        "Measured as a percentage of the frame's area, so it means the same "
        "thing at proxy and at delivery.");
    p->setRange(0.0, 5.0);
    p->setDisplayRange(0.0, 0.5);
    p->setDefault(0.02);
    addParam(page, p, gShape);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamBridgeRadius);
    p->setLabels("Bridge", "Bridge", "Bridge");
    p->setHint(
        "Joins parts that nearly touch, so they break away as one piece. Useful "
        "in Alpha Pieces mode for a script logo whose strokes meet at hairlines. "
        "A percentage of frame height.");
    p->setRange(0.0, 10.0);
    p->setDisplayRange(0.0, 2.0);
    p->setDefault(0.0);
    addParam(page, p, gShape);
  }
  {
    OFX::IntParamDescriptor* p = desc.defineIntParam(shp::kParamShapeSeed);
    p->setLabels("Shape Seed", "Shape Seed", "Shape Seed");
    p->setHint("Changes where the cuts fall without changing anything else.");
    p->setRange(0, 100000);
    p->setDisplayRange(0, 100);
    p->setDefault(1);
    addParam(page, p, gShape);
  }

  // ------------------------------------------------------------------ Forces
  OFX::GroupParamDescriptor* gForce = defineGroup("grpForces", "Forces", true);
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(shp::kParamForceMode);
    p->setLabels("Force Mode", "Force Mode", "Force Mode");
    p->setHint(
        "When pieces let go. All At Once releases everything on the start frame. "
        "Inside Radius Only releases just the pieces a force actually covers, "
        "leaving the rest standing. Radial Wave lets the break spread outward "
        "from the force, which is the look AE gets from an expanding force "
        "sphere.");
    p->appendOption("All At Once");
    p->appendOption("Inside Radius Only");
    p->appendOption("Radial Wave");
    p->setDefault(shp::kForceAllAtStart);
    addParam(page, p, gForce);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamWaveSpeed);
    p->setLabels("Wave Speed", "Wave Speed", "Wave Speed");
    p->setHint(
        "How fast the break front travels outward, as a share of the frame width "
        "per frame.");
    p->setRange(0.001, 10.0);
    p->setDisplayRange(0.005, 0.5);
    p->setDefault(0.05);
    addParam(page, p, gForce);
  }
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(shp::kParamFalloff);
    p->setLabels("Falloff", "Falloff", "Falloff");
    p->setHint("How a force weakens toward the edge of its radius.");
    p->appendOption("Linear");
    p->appendOption("Smooth");
    p->appendOption("Inverse Square");
    p->setDefault(shp::kFalloffSmooth);
    addParam(page, p, gForce);
  }

  struct ForceDef {
    const char* enabled;
    const char* shape;
    const char* pos;
    const char* radius;
    const char* direction;
    const char* width;
    const char* strength;
    const char* delay;
    const char* label;
    bool on;
    double defaultDelay;
  };
  const ForceDef forces[2] = {
      {shp::kParamForce1Enabled, shp::kParamForce1Shape, shp::kParamForce1Pos,
       shp::kParamForce1Radius, shp::kParamForce1Direction, shp::kParamForce1Width,
       shp::kParamForce1Strength, shp::kParamForce1Delay, "Force 1", true, 0.0},
      {shp::kParamForce2Enabled, shp::kParamForce2Shape, shp::kParamForce2Pos,
       shp::kParamForce2Radius, shp::kParamForce2Direction, shp::kParamForce2Width,
       shp::kParamForce2Strength, shp::kParamForce2Delay, "Force 2", false, 12.0},
  };

  for (const ForceDef& f : forces) {
    {
      OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(f.enabled);
      p->setLabels(f.label, f.label, f.label);
      p->setHint("Turn this force on.");
      p->setDefault(f.on);
      addParam(page, p, gForce);
    }
    {
      OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(f.shape);
      p->setLabels("  Shape", "Shape", "Force Shape");
      p->setHint(
          "Radial blows outward from a point and fades with distance from it. "
          "Directional blows along an angle and fades with distance from the "
          "line -- its reach along that line is unlimited, so it behaves like "
          "wind rather than like a blast.");
      p->appendOption("Radial");
      p->appendOption("Directional");
      p->setDefault(shp::kForceRadial);
      addParam(page, p, gForce);
    }
    {
      OFX::Double2DParamDescriptor* p = desc.defineDouble2DParam(f.pos);
      p->setLabels("  Position", "Position", "Position");
      p->setHint(
          "Where the blast comes from, as a fraction of the frame. For a "
          "directional force this is a point on its centre line -- moving it "
          "along the direction changes nothing, moving it across shifts the band.");
      p->setDoubleType(OFX::eDoubleTypePlain);
      p->setRange(-5.0, -5.0, 5.0, 5.0);
      p->setDisplayRange(0.0, 0.0, 1.0, 1.0);
      p->setDefault(0.5, 0.5);
      addParam(page, p, gForce);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(f.radius);
      p->setLabels("  Radius", "Radius", "Radius");
      p->setHint("How far the force reaches, as a fraction of the frame width.");
      p->setRange(0.0, 10.0);
      p->setDisplayRange(0.0, 2.0);
      p->setDefault(0.4);
      addParam(page, p, gForce);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(f.direction);
      p->setLabels("  Direction", "Direction", "Direction");
      p->setHint(
          "Which way it blows, in degrees. Same convention as gravity: 180 is "
          "down, 0 is up, 90 is to the right.");
      p->setRange(-3600.0, 3600.0);
      p->setDisplayRange(0.0, 360.0);
      p->setDefault(90.0);
      addParam(page, p, gForce);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(f.width);
      p->setLabels("  Width", "Width", "Width");
      p->setHint(
          "How wide the band of moving air is, across the direction, as a "
          "fraction of frame width. Pieces near the centre line are thrown "
          "hardest; past the edge nothing moves.");
      p->setRange(0.0, 10.0);
      p->setDisplayRange(0.02, 2.0);
      p->setDefault(0.5);
      addParam(page, p, gForce);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(f.strength);
      p->setLabels("  Strength", "Strength", "Strength");
      p->setHint("How hard it throws. Negative pulls pieces inward instead.");
      p->setRange(-100.0, 100.0);
      p->setDisplayRange(-3.0, 5.0);
      p->setDefault(1.0);
      addParam(page, p, gForce);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(f.delay);
      p->setLabels("  Delay (frames)", "Delay", "Delay (frames)");
      p->setHint(
          "How long after the start frame this force fires. A delayed force "
          "arrives while the pieces are already in the air and shoves them "
          "again, which is a different look from two forces going off together "
          "-- try breaking the frame with one and sweeping it sideways with the "
          "other a few frames later.");
      p->setRange(0.0, 100000.0);
      p->setDisplayRange(0.0, 60.0);
      p->setDefault(f.defaultDelay);
      addParam(page, p, gForce);
    }
  }

  // ----------------------------------------------------------------- Physics
  OFX::GroupParamDescriptor* gPhys = defineGroup("grpPhysics", "Physics", true);
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(shp::kParamSimMode);
    p->setLabels("Simulation", "Simulation", "Simulation");
    p->setHint(
        "Ballistic throws every piece along its own arc and is instant to "
        "scrub -- pieces pass through each other. Rigid Body runs a real "
        "simulation with collisions, a floor and walls; it has to be simulated "
        "forward from the start frame, so the first scrub costs a moment.");
    p->appendOption("Ballistic (fast)");
    p->appendOption("Rigid Body (collisions)");
    p->setDefault(shp::kSimBallistic);
    addParam(page, p, gPhys);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamGravity);
    p->setLabels("Gravity", "Gravity", "Gravity");
    p->setHint("How hard pieces fall. 0 leaves them drifting.");
    p->setRange(-100.0, 100.0);
    p->setDisplayRange(0.0, 20.0);
    p->setDefault(3.0);
    addParam(page, p, gPhys);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamGravityDirection);
    p->setLabels("Gravity Direction", "Gravity Dir", "Gravity Direction");
    p->setHint("Which way is down, in degrees. 180 is straight down; 0 sends pieces up.");
    p->setRange(-3600.0, 3600.0);
    p->setDisplayRange(0.0, 360.0);
    p->setDefault(180.0);
    addParam(page, p, gPhys);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamRotationSpeed);
    p->setLabels("Rotation Speed", "Rotation", "Rotation Speed");
    p->setHint("How fast pieces tumble. Small pieces spin faster than large ones.");
    p->setRange(0.0, 100.0);
    p->setDisplayRange(0.0, 5.0);
    p->setDefault(1.0);
    addParam(page, p, gPhys);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamRandomness);
    p->setLabels("Randomness", "Randomness", "Randomness");
    p->setHint("Scatters speed and spin between pieces so the break looks less mechanical.");
    p->setRange(0.0, 2.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.15);
    addParam(page, p, gPhys);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamViscosity);
    p->setLabels("Viscosity", "Viscosity", "Viscosity");
    p->setHint(
        "Air resistance. Raising it slows pieces toward a drift instead of "
        "letting them accelerate away.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.1);
    addParam(page, p, gPhys);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamMassVariance);
    p->setLabels("Mass Variance", "Mass Var", "Mass Variance");
    p->setHint("Randomises how heavy each piece is, so they do not all fly at the same speed.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.3);
    addParam(page, p, gPhys);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(shp::kParamCollideShards);
    p->setLabels("Piece Collisions", "Collide", "Piece Collisions");
    p->setHint("Let pieces hit each other, not just the floor. The expensive part of the sim.");
    p->setDefault(true);
    addParam(page, p, gPhys);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamBounciness);
    p->setLabels("Bounciness", "Bounce", "Bounciness");
    p->setHint("How much speed a piece keeps in a collision.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.3);
    addParam(page, p, gPhys);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamFriction);
    p->setLabels("Friction", "Friction", "Friction");
    p->setHint("How much pieces grip when they slide against each other or the floor.");
    p->setRange(0.0, 2.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.4);
    addParam(page, p, gPhys);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(shp::kParamFloorEnabled);
    p->setLabels("Floor", "Floor", "Floor");
    p->setHint("Give pieces something to land on.");
    p->setDefault(true);
    addParam(page, p, gPhys);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamFloorY);
    p->setLabels("  Floor Height", "Floor Y", "Floor Height");
    p->setHint("Where the floor sits, as a fraction down the frame. 1 is the bottom edge.");
    p->setRange(-2.0, 3.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(1.0);
    addParam(page, p, gPhys);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(shp::kParamWallsEnabled);
    p->setLabels("Walls", "Walls", "Walls");
    p->setHint("Keep pieces inside the frame by bouncing them off the left and right edges.");
    p->setDefault(false);
    addParam(page, p, gPhys);
  }
  {
    OFX::IntParamDescriptor* p = desc.defineIntParam(shp::kParamSubsteps);
    p->setLabels("Accuracy", "Accuracy", "Accuracy");
    p->setHint(
        "Simulation steps per frame. Raise it if fast pieces are passing through "
        "each other or through the floor; lower it if the sim is slow.");
    p->setRange(1, 16);
    p->setDisplayRange(1, 8);
    p->setDefault(4);
    addParam(page, p, gPhys);
  }

  // ------------------------------------------------------------------ Timing
  OFX::GroupParamDescriptor* gTime = defineGroup("grpTiming", "Timing", true);
  {
    OFX::IntParamDescriptor* p = desc.defineIntParam(shp::kParamStartFrame);
    p->setLabels("Start (frames)", "Start", "Start (frames)");
    p->setHint(
        "Which frame of the CLIP the shatter begins on -- 0 is the clip's first "
        "frame, so trimming or moving the clip carries the timing with it. "
        "Frames rather than seconds, deliberately: the frame rate is not "
        "something every host will tell a plugin.");
    p->setRange(-100000, 100000);
    p->setDisplayRange(0, 120);
    p->setDefault(0);
    addParam(page, p, gTime);
  }
  {
    OFX::IntParamDescriptor* p = desc.defineIntParam(shp::kParamSeed);
    p->setLabels("Motion Seed", "Motion Seed", "Motion Seed");
    p->setHint("Re-rolls how the pieces fly without changing where they were cut.");
    p->setRange(0, 100000);
    p->setDisplayRange(0, 100);
    p->setDefault(1);
    addParam(page, p, gTime);
  }

  // ------------------------------------------------------------------ Render
  OFX::GroupParamDescriptor* gRender = defineGroup("grpRender", "Render", false);
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamFadeDuration);
    p->setLabels("Fade Out (frames)", "Fade", "Fade Out (frames)");
    p->setHint("How long a piece takes to fade away once it starts. 0 means pieces never fade.");
    p->setRange(0.0, 10000.0);
    p->setDisplayRange(0.0, 60.0);
    p->setDefault(0.0);
    addParam(page, p, gRender);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamFadeStart);
    p->setLabels("  Fade Delay", "Fade Delay", "Fade Delay");
    p->setHint("How long a piece stays solid after it detaches, before the fade begins.");
    p->setRange(0.0, 10000.0);
    p->setDisplayRange(0.0, 60.0);
    p->setDefault(0.0);
    addParam(page, p, gRender);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(shp::kParamShrinkOnFade);
    p->setLabels("  Shrink While Fading", "Shrink", "Shrink While Fading");
    p->setHint("Pieces also scale down as they fade, so they read as receding rather than dissolving.");
    p->setDefault(false);
    addParam(page, p, gRender);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(shp::kParamMotionBlur);
    p->setLabels("Motion Blur", "Motion Blur", "Motion Blur");
    p->setHint(
        "Smears each piece along the way it is travelling. Costs one extra "
        "sample per piece per step, and only pieces that are actually moving "
        "pay for it.");
    p->setDefault(false);
    addParam(page, p, gRender);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(shp::kParamShutterAngle);
    p->setLabels("  Shutter Angle", "Shutter", "Shutter Angle");
    p->setHint("How long the shutter stays open. 180 degrees is half a frame, the film default.");
    p->setRange(0.0, 720.0);
    p->setDisplayRange(0.0, 360.0);
    p->setDefault(180.0);
    addParam(page, p, gRender);
  }
  {
    OFX::IntParamDescriptor* p = desc.defineIntParam(shp::kParamBlurSamples);
    p->setLabels("  Blur Samples", "Samples", "Blur Samples");
    p->setHint("More samples make a smoother smear. Past eight the difference stops showing.");
    p->setRange(2, 8);
    p->setDisplayRange(2, 8);
    p->setDefault(8);
    addParam(page, p, gRender);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(shp::kParamEdgeAntialias);
    p->setLabels("Smooth Edges", "Smooth Edges", "Smooth Edges");
    p->setHint(
        "Softens the cut edges of each piece. Worth turning off only if you are "
        "chasing render time.");
    p->setDefault(true);
    addParam(page, p, gRender);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(shp::kParamShowPieces);
    p->setLabels("Show Pieces", "Show Pieces", "Show Pieces");
    p->setHint(
        "Tint each piece a different colour. Turn this on first: it shows what "
        "the shatter is actually going to break apart, before worrying about how "
        "it moves.");
    p->setDefault(false);
    addParam(page, p, gRender);
  }
}

OFX::ImageEffect* ShatterPhysicsFactory::createInstance(OfxImageEffectHandle handle,
                                                        OFX::ContextEnum) {
  return new ShatterPhysicsPlugin(handle);
}

}  // namespace

namespace OFX {
namespace Plugin {

void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static ShatterPhysicsFactory factory(kPluginIdentifier, 1, 0);
  ids.push_back(&factory);
}

}  // namespace Plugin
}  // namespace OFX
