// Mode A: where a shard is at time t, in closed form.
//
// No state, no stepping, no cache. Given the frame number this evaluates the
// exact position of every piece directly, which is what makes it safe under a
// host that renders frame 200 before frame 3 and keeps whichever frames it
// likes. It is also what makes scrubbing instant.
//
// The price is that pieces pass through each other and through the ground.
// Mode B (Physics2D + SimCache) buys that back at the cost of having to simulate
// forward from the start.
#pragma once

#include "Forces.h"
#include "ShardGeom.h"

namespace shp {

/** @brief Integrate constant gravity plus linear drag, exactly.
 *
 *   v' = g - k*v      =>   v(t) = (v0 - g/k) e^{-kt} + g/k
 *                          p(t) = p0 + (v0 - g/k)(1 - e^{-kt})/k + (g/k) t
 *
 * The drag-free branch is not an optimisation. At k = 0 the expressions above
 * are 0/0, and in floats they go visibly wrong long before k reaches zero -- so
 * the plain ballistic form takes over while it is still the more accurate of
 * the two.
 */
SHP_HD inline void IntegrateDrag(const Vec2& v0, const Vec2& g, float k, float t,
                                 Vec2& outPos, Vec2& outVel) {
  if (k < 1e-4f) {
    outPos = v0 * t + g * (0.5f * t * t);
    outVel = v0 + g * t;
    return;
  }
  const float invK = 1.0f / k;
  const Vec2 terminal = g * invK;          // the speed drag settles at
  const Vec2 excess = v0 - terminal;       // what decays away
  const float decay = expf(-k * t);

  outPos = excess * ((1.0f - decay) * invK) + terminal * t;
  outVel = excess * decay + terminal;
}

/** @brief The scalar twin of IntegrateDrag, for rotation. */
SHP_HD inline void IntegrateSpin(float w0, float k, float t, float& outAngle, float& outW) {
  if (k < 1e-4f) {
    outAngle = w0 * t;
    outW = w0;
    return;
  }
  const float invK = 1.0f / k;
  const float decay = expf(-k * t);
  outAngle = w0 * (1.0f - decay) * invK;
  outW = w0 * decay;
}

/** @brief Fade and shrink, applied after the motion.
 *
 * Deliberately a separate step from the simulation rather than a term inside it:
 * these are render controls, and folding them in would mean that nudging the
 * fade length invalidated a cached rigid-body run.
 *
 * @param age frames since this shard detached.
 */
SHP_HD inline void ApplyFade(float age, float fadeStart, float fadeDuration, bool shrink,
                             float& outAlpha, float& outScale) {
  outAlpha = 1.0f;
  outScale = 1.0f;
  if (fadeDuration <= 0.0f) return;   // 0 means never fade

  const float t = (age - fadeStart) / fadeDuration;
  if (t <= 0.0f) return;

  const float f = t >= 1.0f ? 1.0f : t;
  outAlpha = 1.0f - f;
  if (shrink) {
    // Stops at 5% rather than 0: a shard scaled to nothing inverts to an
    // enormous sampling footprint, and the transform's degenerate-matrix guard
    // would hand back identity, briefly flashing the piece at full size on the
    // last frame before it vanishes.
    outScale = 1.0f - 0.95f * f;
  }
}

struct RenderParams {
  float fadeStart = 0.0f;
  /// 0 disables fading entirely, which is the default because AE's Shatter does
  /// not fade either -- pieces leave the frame rather than dissolving in it. A
  /// nonzero default is worse than it sounds: it makes every shatter look weak
  /// without the user having asked for anything.
  float fadeDuration = 0.0f;
  bool shrinkOnFade = false;
};

/** @brief Evaluate every shard's pose at clip time @p t, in frames.
 *
 * @param out must already be sized to the shard count.
 */
inline void EvaluateBallistic(const std::vector<Shard>& shards, const ForceParams& fp,
                              const PhysicsParams& pp, const RenderParams& rp,
                              const FrameScale& fs, float t, std::vector<ShardPose>& out) {
  out.assign(shards.size(), ShardPose{});

  const Vec2 g = GravityVector(pp, fs);
  const float drag = pp.viscosity * kDragScale;

  for (std::size_t i = 0; i < shards.size(); ++i) {
    const Shard& sh = shards[i];
    ShardPose& p = out[i];

    ShardSim ss;
    ss.cx = sh.cx;
    ss.cy = sh.cy;
    ss.area = sh.area;
    ss.radius = sh.radius;
    ss.inertiaPerMass = sh.inertiaPerMass;

    const float detach = DetachFrame(ss, fp, fs);
    const float age = t - detach;
    if (age < 0.0f) continue;   // still at rest: pose stays all-zero, active false

    const uint32_t id = uint32_t(sh.id);
    const float mass = MassOf(ss, fs, pp, id, uint32_t(fp.seed));
    const Vec2 v0 = InitialVelocity(ss, fp, pp, fs, id, mass);
    const float w0 = InitialSpin(ss, fp, pp, fs, id, mass);

    Vec2 pos, vel;
    IntegrateDrag(v0, g, drag, age, pos, vel);

    float angle = 0.0f, w = 0.0f;
    IntegrateSpin(w0, drag, age, angle, w);

    p.active = true;
    p.x = pos.x;
    p.y = pos.y;
    p.vx = vel.x;
    p.vy = vel.y;
    p.angle = angle;
    p.w = w;
    ApplyFade(age, rp.fadeStart, rp.fadeDuration, rp.shrinkOnFade, p.alpha, p.scale);
  }
}

}  // namespace shp
