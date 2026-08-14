// What makes a piece move, and when it starts.
//
// Both simulation modes share this file. That is deliberate and load-bearing:
// the rigid-body mode launches each shard with exactly the initial conditions
// the ballistic mode would have given it, so switching modes changes how the
// pieces *interact*, not where they were thrown. Without that, toggling the mode
// would look like a different effect rather than the same effect with collisions
// turned on.
//
// Every quantity here is in frames and pixels: velocity is px/frame, gravity is
// px/frame^2, spin is rad/frame. There is no seconds anywhere, for the reason
// spelled out in ClipTime.h -- Fusion does not publish a frame rate and asking
// for one throws.
#pragma once

#include "Hash.h"
#include "TransformMath.h"

namespace shp {

enum class ForceMode {
  AllAtStart   = 0,  ///< everything lets go on the start frame
  InsideRadius = 1,  ///< only pieces a force actually reaches ever move
  RadialWave   = 2,  ///< the break spreads outward from the force
};

enum class Falloff {
  Linear        = 0,
  Smooth        = 1,
  InverseSquare = 2,
};

enum class ForceShape {
  Radial      = 0,  ///< blows outward from a point, inside a radius
  Directional = 1,  ///< blows along an angle, inside a band of some width
};

struct ForceField {
  bool enabled = false;
  ForceShape shape = ForceShape::Radial;
  float posX = 0.5f, posY = 0.5f;  ///< normalized [0,1] within the frame

  float radius = 0.4f;             ///< Radial only. Fraction of frame WIDTH.

  /// Directional only. Degrees, same convention as gravity: 180 is down, 0 is
  /// up, 90 is to the right.
  float direction = 90.0f;
  /// Directional only. Full width of the band, as a fraction of frame width.
  /// Reach along the direction is unlimited -- it is a wind, not a puff.
  float width = 0.5f;

  float strength = 1.0f;

  /// Frames after the shatter's start before this force fires.
  ///
  /// This is what lets one force break the frame and a second one, arriving
  /// later, shove the pieces mid-flight. Both simulation modes handle it as a
  /// genuine second impulse rather than as a blend of two launches.
  float delay = 0.0f;
};

struct ForceParams {
  ForceField f1;
  ForceField f2;
  ForceMode mode = ForceMode::AllAtStart;
  float waveSpeed = 0.05f;  ///< fraction of frame width per frame
  Falloff falloff = Falloff::Smooth;
  int startFrame = 0;
  int seed = 1;
};

struct PhysicsParams {
  float rotationSpeed = 1.0f;
  float randomness = 0.15f;
  float viscosity = 0.1f;
  float massVariance = 0.3f;
  float gravity = 3.0f;
  float gravityDirection = 180.0f;  ///< degrees; 180 is straight down

  // Rigid-body only. Ignored entirely in ballistic mode, and deliberately still
  // part of this struct so that the simulation cache key covers them.
  float bounciness = 0.3f;
  float friction = 0.4f;
  bool collideShards = true;
  bool floorEnabled = true;
  float floorY = 1.0f;  ///< fraction of frame height
  bool wallsEnabled = false;
  int substeps = 4;
};

// --------------------------------------------------------------- unit scales
//
// These turn the dimensionless dials the user sees into pixels per frame. They
// exist so that "Gravity 3" means something reasonable at any resolution, and
// they are expressed against the frame size rather than as absolute pixels for
// the resolution-independence reason in Cells.h.
//
// The numbers were chosen so that the defaults produce a shatter that clears a
// 1080p frame in roughly a second of 24fps footage. They are not physical; AE's
// Shatter is not either.

/// Strength 1.0 at the centre of a force, on a shard of average mass.
constexpr float kVelocityPerStrength = 0.015f;   // x frame width, px/frame
/// Gravity 1.0, per frame squared.
constexpr float kGravityScale = 0.0005f;         // x frame height, px/frame^2
/// Rotation Speed 1.0, on a shard of average size.
constexpr float kSpinScale = 0.05f;              // rad/frame
/// Viscosity 1.0 as a linear drag coefficient. At 0.15/frame a shard loses half
/// its excess speed in about five frames, which reads as air rather than as syrup.
constexpr float kDragScale = 0.15f;

// ------------------------------------------------------------------ geometry
//
// The per-shard facts the simulation needs, pulled out of Shard so that this
// header stays free of std::vector and can be compiled as device code later.

struct ShardSim {
  float cx = 0, cy = 0;
  float area = 0;
  float radius = 0;
  float inertiaPerMass = 0;
};

/// Frame-level constants derived once per render.
struct FrameScale {
  float width = 1920.0f;
  float height = 1080.0f;
  float meanArea = 1.0f;
  float meanRadius = 1.0f;
};

SHP_HD inline float ApplyFalloff(float t, Falloff mode) {
  // t is 0 at the force's centre and 1 at the edge of its radius.
  if (t >= 1.0f) return 0.0f;
  if (t <= 0.0f) return 1.0f;
  switch (mode) {
    case Falloff::Linear:
      return 1.0f - t;
    case Falloff::InverseSquare:
      // Softened at the centre: a true 1/r^2 is unbounded there, and one shard
      // landing a pixel from the force point would leave the frame in a frame.
      return 1.0f / (1.0f + 9.0f * t * t) - 0.1f;
    case Falloff::Smooth:
    default: {
      const float u = 1.0f - t;
      return u * u * (3.0f - 2.0f * u);  // smoothstep
    }
  }
}

/** @brief A shard's mass, relative to the frame's average shard.
 *
 * Square root of the area rather than the area itself: a piece four times the
 * size of another is heavier, but not four times as sluggish, and the linear
 * version makes big pieces look nailed down while confetti flies. AE's Mass
 * Variance dial does something similar. */
SHP_HD inline float MassOf(const ShardSim& s, const FrameScale& fs, const PhysicsParams& pp,
                           uint32_t shardId, uint32_t seed) {
  const float rel = fs.meanArea > 0.0f ? s.area / fs.meanArea : 1.0f;
  float m = sqrtf(rel > 0.0f ? rel : 1.0f);
  m *= 1.0f + pp.massVariance * HashSigned(shardId, seed, kChanMass);
  return m < 0.05f ? 0.05f : m;
}

/** @brief What one force does to one shard, and when.
 *
 * Forces are impulses with a time, not a single blended launch velocity. That
 * distinction is what makes a delayed second force interesting: it arrives while
 * the piece is already in the air and changes its trajectory, rather than being
 * averaged into how it left in the first place. */
struct ForceImpulse {
  bool hits = false;      ///< does this force reach this shard at all
  float time = 1e9f;      ///< absolute clip frame it arrives on
  Vec2 linear{0.0f, 0.0f};///< added to velocity, px/frame, already mass-divided
};

/** @brief Unit vector for an angle in the gravity convention: 180 is down. */
SHP_HD inline Vec2 AngleToVector(float degrees) {
  const float a = degrees * kPi / 180.0f;
  return Vec2(sinf(a), -cosf(a));
}

/** @brief The purely geometric half of a force: whether it reaches a shard,
 *  how hard, which way, and when.
 *
 * Split out from the impulse deliberately. Timing depends only on geometry --
 * not on the shard's mass, and not on the randomness draw -- so detach times
 * stay stable when someone drags Mass Variance, and the solver can ask "when"
 * without having to first work out "how hard".
 */
struct ForceReach {
  bool hits = false;
  float k = 0.0f;          ///< falloff, 1 at the centre of the force, 0 at its edge
  Vec2 dir{0.0f, 0.0f};    ///< unit push direction
  float time = 1e9f;       ///< absolute clip frame it arrives on
};

/** @brief Where and when one force touches one shard.
 *
 * Radial forces fall off with distance from a POINT and push outward from it.
 * Directional forces fall off with distance from a LINE and push along it --
 * reach along the line is unlimited, because a wind that stops after a few
 * hundred pixels is not what anyone means by a directional force.
 */
SHP_HD inline ForceReach ReachOf(const ForceField& f, const ForceParams& fp,
                                 const FrameScale& fs, const ShardSim& s, uint32_t shardId) {
  ForceReach out;
  if (!f.enabled) return out;

  const Vec2 origin(f.posX * fs.width, f.posY * fs.height);
  const Vec2 rel(s.cx - origin.x, s.cy - origin.y);
  float waveDist = 0.0f;

  if (f.shape == ForceShape::Directional) {
    out.dir = AngleToVector(f.direction);

    // Falloff is across the band, measured perpendicular to the axis.
    const Vec2 perp(-out.dir.y, out.dir.x);
    const float half = f.width * fs.width * 0.5f;
    if (half <= 1e-6f) return out;
    out.k = ApplyFalloff(fabsf(Dot(rel, perp)) / half, fp.falloff);

    // For the wave, the front travels ALONG the direction. Anything at or
    // behind the axis line has already been passed, so it sits at zero -- the
    // alternative, a negative distance, would have upwind pieces detaching
    // before the shatter had started.
    const float along = Dot(rel, out.dir);
    waveDist = along > 0.0f ? along : 0.0f;
  } else {
    const float r = f.radius * fs.width;
    if (r <= 1e-6f) return out;

    const float dist = Length(rel);
    out.k = ApplyFalloff(dist / r, fp.falloff);
    waveDist = dist;

    // A shard sitting exactly on the force point has no direction to go, so it
    // is given one from its own hash rather than left as the only piece that
    // never moves.
    out.dir = Normalize(rel);
    if (out.dir.x == 0.0f && out.dir.y == 0.0f) {
      const float a = Hash01(shardId, uint32_t(fp.seed), kChanVelAngle) * 2.0f * kPi;
      out.dir = Vec2(cosf(a), sinf(a));
    }
  }

  if (out.k <= 0.0f) return out;

  out.hits = true;
  out.time = float(fp.startFrame) + f.delay;

  if (fp.mode == ForceMode::RadialWave) {
    const float speed = fp.waveSpeed * fs.width;
    // A stalled wave never arrives anywhere, which is more honest than
    // arriving everywhere at once.
    if (speed <= 1e-6f) {
      out.hits = false;
      return out;
    }
    out.time += waveDist / speed;
  }
  return out;
}

/** @brief The velocity one force adds to one shard, and when. */
SHP_HD inline ForceImpulse EvaluateForce(const ForceField& f, const ForceParams& fp,
                                         const PhysicsParams& pp, const FrameScale& fs,
                                         const ShardSim& s, uint32_t shardId, float mass) {
  ForceImpulse out;
  const ForceReach r = ReachOf(f, fp, fs, s, shardId);
  if (!r.hits) return out;

  Vec2 v = r.dir * (f.strength * r.k * kVelocityPerStrength * fs.width);

  // Randomness scatters both direction and magnitude. Scaling the jitter by the
  // speed this impulse already carries keeps a gentle shatter gentle -- an
  // absolute jitter would dominate whenever the forces were weak, and every
  // piece would fly off at the same speed regardless of the strength dial.
  if (pp.randomness > 0.0f) {
    const float speed = Length(v);
    const float a = Hash01(shardId, uint32_t(fp.seed), kChanVelAngle) * 2.0f * kPi;
    const float m = Hash01(shardId, uint32_t(fp.seed), kChanVelMag);
    v += Vec2(cosf(a), sinf(a)) * (speed * pp.randomness * m);
  }

  out.hits = true;
  out.linear = v * (1.0f / mass);
  out.time = r.time;
  return out;
}

/** @brief Both forces, in the order they arrive.
 *
 * @param out must have room for two entries. Returns how many actually hit.
 *
 * Sorted by time so callers can walk them as a timeline without re-checking.
 * Two entries makes that a compare-and-swap rather than a sort.
 */
SHP_HD inline int GatherImpulses(const ShardSim& s, const ForceParams& fp,
                                 const PhysicsParams& pp, const FrameScale& fs,
                                 uint32_t shardId, float mass, ForceImpulse* out) {
  const ForceImpulse a = EvaluateForce(fp.f1, fp, pp, fs, s, shardId, mass);
  const ForceImpulse b = EvaluateForce(fp.f2, fp, pp, fs, s, shardId, mass);

  int n = 0;
  if (a.hits) out[n++] = a;
  if (b.hits) out[n++] = b;
  if (n == 2 && out[1].time < out[0].time) {
    const ForceImpulse t = out[0];
    out[0] = out[1];
    out[1] = t;
  }
  return n;
}

/** @brief The frame this shard lets go on.
 *
 * A very large value means never. Returning a float rather than an int lets the
 * radial wave arrive between frames, so the break front advances smoothly
 * instead of in visible steps.
 *
 * A piece can detach BEFORE any force reaches it: in All At Once everything lets
 * go on the start frame, so a delayed force finds pieces that have already been
 * falling for a while. That is the whole point of the delay.
 */
SHP_HD inline float DetachFrame(const ShardSim& s, const ForceParams& fp, const FrameScale& fs) {
  // Everything lets go at once, whether or not a force ever reaches it. Pieces
  // outside every force simply fall.
  if (fp.mode == ForceMode::AllAtStart) return float(fp.startFrame);

  // Otherwise a piece only moves once something actually arrives, and it moves
  // when the FIRST thing arrives. Only the geometry matters here, which is why
  // this can ask ReachOf directly instead of building impulses it would discard.
  const ForceReach a = ReachOf(fp.f1, fp, fs, s, uint32_t(s.area));
  const ForceReach b = ReachOf(fp.f2, fp, fs, s, uint32_t(s.area));

  float best = 1e9f;
  if (a.hits && a.time < best) best = a.time;
  if (b.hits && b.time < best) best = b.time;
  return best;
}

/** @brief Initial angular velocity, rad/frame.
 *
 * Small pieces spin faster than large ones, which is both true of real debris
 * and the single cue that most makes a shatter read as physical rather than as
 * a slide. */
SHP_HD inline float InitialSpin(const ShardSim& s, const ForceParams& fp, const PhysicsParams& pp,
                                const FrameScale& fs, uint32_t shardId, float mass) {
  const float sizeRatio = s.radius > 1e-3f ? fs.meanRadius / s.radius : 1.0f;
  const float clamped = Clampf(sizeRatio, 0.25f, 4.0f);

  float w = pp.rotationSpeed * kSpinScale * clamped;
  w *= HashSigned(shardId, uint32_t(fp.seed), kChanSpin);        // direction and magnitude
  w *= 1.0f + pp.randomness * HashSigned(shardId, uint32_t(fp.seed), kChanSpinJitter);
  return w / mass;
}

/** @brief Gravity as a pixel-space vector.
 *
 * 180 degrees is straight down. Core code is TOP-DOWN -- row 0 is the top and +y
 * goes downward -- so "down" is +y here, and the plugin's view builder is what
 * reconciles that with OFX's bottom-up buffers. Getting this sign wrong is
 * invisible on a symmetric test image, which is why the harness holds a
 * deliberately asymmetric one. */
SHP_HD inline Vec2 GravityVector(const PhysicsParams& pp, const FrameScale& fs) {
  const float a = pp.gravityDirection * kPi / 180.0f;
  const float g = pp.gravity * kGravityScale * fs.height;
  return Vec2(sinf(a) * g, -cosf(a) * g);
}

}  // namespace shp
