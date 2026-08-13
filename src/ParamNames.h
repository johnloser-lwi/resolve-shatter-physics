// Every parameter name, in one place.
//
// These strings are an on-disk contract, not labels. The host persists a
// parameter by name into the project file and a preset stores it by name, so
// renaming one here silently orphans every saved setting that used it: the
// project loads, the parameter reverts to its default, and nothing reports an
// error. Labels are free to change; these are not.
//
// The same applies to the *order* of choice options, which the host persists as
// an integer index. The enums below spell those indices out rather than leaving
// them implicit in the appendOption call order, so that inserting an option in
// the middle is a visible edit here rather than a silent reinterpretation of
// every project that used the options after it.
#pragma once

namespace shp {

// ------------------------------------------------------------------- Shape
constexpr const char* kParamPattern        = "pattern";
constexpr const char* kParamCellCount      = "cellCount";
constexpr const char* kParamCellJitter     = "cellJitter";
constexpr const char* kParamBrickRows      = "brickRows";
constexpr const char* kParamBrickOffset    = "brickOffset";
constexpr const char* kParamMapLevels      = "mapLevels";
constexpr const char* kParamAlphaThreshold = "alphaThreshold";
constexpr const char* kParamMinPieceArea   = "minPieceArea";
constexpr const char* kParamBridgeRadius   = "bridgeRadius";
constexpr const char* kParamShapeSeed      = "shapeSeed";

// Pattern choice indices.
enum PatternIndex {
  kPatternVoronoi = 0,   // "Glass" in AE terms
  kPatternBricks  = 1,
  kPatternGrid    = 2,
  kPatternAlpha   = 3,   // one connected alpha region = one piece
  kPatternMap     = 4,   // custom shatter map on the second input
};

// ------------------------------------------------------------------ Forces
constexpr const char* kParamForce1Enabled  = "force1Enabled";
constexpr const char* kParamForce1Pos      = "force1Pos";
constexpr const char* kParamForce1Radius   = "force1Radius";
constexpr const char* kParamForce1Strength = "force1Strength";
constexpr const char* kParamForce2Enabled  = "force2Enabled";
constexpr const char* kParamForce2Pos      = "force2Pos";
constexpr const char* kParamForce2Radius   = "force2Radius";
constexpr const char* kParamForce2Strength = "force2Strength";
constexpr const char* kParamForceMode      = "forceMode";
constexpr const char* kParamWaveSpeed      = "waveSpeed";
constexpr const char* kParamFalloff        = "falloff";

// AE's Force block is 3D: a force has a Depth, and a piece detaches when the
// expanding sphere reaches it. In 2D the depth axis is gone, so the useful part
// of that behaviour -- pieces leaving in a wave rather than all at once -- is
// exposed directly as a mode instead.
enum ForceModeIndex {
  kForceAllAtStart    = 0,  // everything detaches on the start frame
  kForceInsideRadius  = 1,  // only pieces within some force radius ever move
  kForceRadialWave    = 2,  // detach time grows with distance from the force
};

enum FalloffIndex {
  kFalloffLinear        = 0,
  kFalloffSmooth        = 1,
  kFalloffInverseSquare = 2,
};

// ----------------------------------------------------------------- Physics
constexpr const char* kParamSimMode          = "simMode";
constexpr const char* kParamRotationSpeed    = "rotationSpeed";
constexpr const char* kParamRandomness       = "randomness";
constexpr const char* kParamViscosity        = "viscosity";
constexpr const char* kParamMassVariance     = "massVariance";
constexpr const char* kParamGravity          = "gravity";
constexpr const char* kParamGravityDirection = "gravityDirection";
constexpr const char* kParamBounciness       = "bounciness";
constexpr const char* kParamFriction         = "friction";
constexpr const char* kParamCollideShards    = "collideShards";
constexpr const char* kParamFloorEnabled     = "floorEnabled";
constexpr const char* kParamFloorY           = "floorY";
constexpr const char* kParamWallsEnabled     = "wallsEnabled";
constexpr const char* kParamSubsteps         = "substeps";

enum SimModeIndex {
  kSimBallistic = 0,   // closed form, evaluated at any t, no collisions
  kSimRigidBody = 1,   // stepped simulation with contacts, cached
};

// ------------------------------------------------------------------ Timing
constexpr const char* kParamStartFrame = "startFrame";
constexpr const char* kParamSeed       = "seed";

// ------------------------------------------------------------------ Render
constexpr const char* kParamFadeStart     = "fadeStart";
constexpr const char* kParamFadeDuration  = "fadeDuration";
constexpr const char* kParamShrinkOnFade  = "shrinkOnFade";
constexpr const char* kParamEdgeAntialias = "edgeAntialias";
constexpr const char* kParamMotionBlur    = "motionBlur";
constexpr const char* kParamShutterAngle  = "shutterAngle";
constexpr const char* kParamBlurSamples   = "blurSamples";
constexpr const char* kParamShowPieces    = "showPieces";

// --------------------------------------------------------------- UI only
// Registered for presets but never invalidating the render cache.
constexpr const char* kParamStatus = "status";

// ------------------------------------------------------------------- Clips
constexpr const char* kClipShatterMap = "ShatterMap";

}  // namespace shp
