// A third axis, in the only two places that can see one: how big a piece draws,
// and which piece draws on top.
//
// There is no 3D here and deliberately so. The simulation stays flat -- forces,
// gravity, collisions and the solver all work in the plane, which is what keeps
// the rigid-body mode tractable and the ballistic mode closed-form. What gets
// added is a per-piece z that nothing collides against and that only the
// compositor reads.
//
// That buys three of the four cues an eye actually uses for depth:
//
//   perspective   near pieces are larger, far ones smaller
//   occlusion     near pieces cover far ones
//   aerial        far pieces are dimmer and lower contrast
//
// The fourth, binocular disparity, is not available on a flat screen anyway. The
// remaining giveaway is that a piece is a flat cutout, which is what the bevel
// in EdgeField.h and the tumble below are for.
#pragma once

#include "Forces.h"
#include "ShardGeom.h"

namespace shp {

struct DepthParams {
  /// How much z velocity pieces pick up, as a fraction of the speed they were
  /// thrown at. 0 keeps everything in the plane and turns the whole feature off.
  float spread = 0.0f;

  /// Pushes everything toward (negative) or away from (positive) the camera,
  /// independent of the forces. Useful on its own for a piece of artwork that
  /// should recede as it breaks.
  float bias = 0.0f;

  /// Strength of the perspective divide. 0 is orthographic -- z then affects
  /// only draw order and shading, which is occasionally what you want.
  float perspective = 0.5f;

  /// How much dimmer a piece gets as it recedes.
  float aerial = 0.0f;

  /// How hard pieces flip about an in-plane axis. Shares Rotation Speed's
  /// timing, so a piece that spins fast also flips fast.
  float tumble = 0.0f;
};

/// Scale factor for a piece at depth @p z.
///
/// A pinhole divide, with the focal length expressed against the frame width so
/// the same setting means the same thing at any resolution. Clamped well away
/// from the singularity at z = -focal: a piece that flies past the camera should
/// grow large and leave, not invert through infinity.
SHP_HD inline float PerspectiveScale(float z, const DepthParams& dp, const FrameScale& fs) {
  if (dp.perspective <= 0.0f) return 1.0f;
  const float focal = fs.width * (1.0f / Clampf(dp.perspective, 0.001f, 4.0f));
  const float denom = focal + z;
  if (denom < focal * 0.05f) return 20.0f;
  return Clampf(focal / denom, 0.02f, 20.0f);
}

/// Brightness multiplier for a piece at depth @p z. Never brightens: a piece in
/// front of the plane is already selling itself by being bigger.
SHP_HD inline float AerialShade(float z, const DepthParams& dp, const FrameScale& fs) {
  if (dp.aerial <= 0.0f || z <= 0.0f) return 1.0f;
  const float t = Clamp01(z / (fs.width * 1.5f));
  return Clampf(1.0f - dp.aerial * t, 0.0f, 1.0f);
}

/// Initial z velocity for a piece, px/frame.
///
/// Proportional to how hard the piece was actually thrown, so a gentle shatter
/// stays shallow and a violent one scatters through depth. An absolute value
/// would make every shatter equally deep regardless of its force.
SHP_HD inline float InitialDepthVelocity(const DepthParams& dp, float planarSpeed,
                                         const FrameScale& fs, uint32_t shardId,
                                         uint32_t seed) {
  if (dp.spread <= 0.0f && dp.bias == 0.0f) return 0.0f;
  const float scatter = planarSpeed * dp.spread * HashSigned(shardId, seed, kChanDepth);
  return scatter + dp.bias * kVelocityPerStrength * fs.width;
}

/// The in-plane axis a piece flips about, and how fast. Drawn from its own hash
/// channels so the flip is uncorrelated with the spin -- a piece doing both at
/// once is what reads as tumbling rather than as a turntable.
SHP_HD inline void InitialTumble(const DepthParams& dp, const PhysicsParams& pp,
                                 uint32_t shardId, uint32_t seed, float& outRate,
                                 float& outAxis) {
  outAxis = Hash01(shardId, seed, kChanTumbleAxis) * 2.0f * kPi;
  outRate = dp.tumble * pp.rotationSpeed * kSpinScale * 2.0f *
            HashSigned(shardId, seed, kChanTumbleRate);
}

}  // namespace shp
