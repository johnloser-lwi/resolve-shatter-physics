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

struct ForceField {
  bool enabled = false;
  float posX = 0.5f, posY = 0.5f;  ///< normalized [0,1] within the frame
  float radius = 0.4f;             ///< fraction of frame WIDTH
  float strength = 1.0f;
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

/** @brief Distance in pixels from a shard to the nearer of the enabled forces.
 *
 * Returns a negative number when no force is enabled, which callers read as
 * "nothing is going to reach this shard". */
SHP_HD inline float NearestForceDistance(const ShardSim& s, const ForceParams& fp,
                                         const FrameScale& fs) {
  float best = -1.0f;
  const ForceField* fields[2] = {&fp.f1, &fp.f2};
  for (int i = 0; i < 2; ++i) {
    if (!fields[i]->enabled) continue;
    const float px = fields[i]->posX * fs.width;
    const float py = fields[i]->posY * fs.height;
    const float d = Length(Vec2(s.cx - px, s.cy - py));
    if (best < 0.0f || d < best) best = d;
  }
  return best;
}

/** @brief The frame this shard lets go on.
 *
 * A very large value means never. Returning a float rather than an int lets the
 * radial wave arrive between frames, so the break front advances smoothly
 * instead of in visible steps. */
SHP_HD inline float DetachFrame(const ShardSim& s, const ForceParams& fp, const FrameScale& fs) {
  const float start = float(fp.startFrame);

  switch (fp.mode) {
    case ForceMode::InsideRadius: {
      // Only pieces some force actually covers ever detach. Everything else
      // stays exactly where it was drawn, which is what makes this mode useful
      // for punching a hole through something and leaving the rest standing.
      const ForceField* fields[2] = {&fp.f1, &fp.f2};
      for (int i = 0; i < 2; ++i) {
        if (!fields[i]->enabled) continue;
        const float px = fields[i]->posX * fs.width;
        const float py = fields[i]->posY * fs.height;
        const float r = fields[i]->radius * fs.width;
        if (LengthSq(Vec2(s.cx - px, s.cy - py)) <= r * r) return start;
      }
      return 1e9f;
    }

    case ForceMode::RadialWave: {
      const float d = NearestForceDistance(s, fp, fs);
      if (d < 0.0f) return 1e9f;
      const float speed = fp.waveSpeed * fs.width;
      if (speed <= 1e-6f) return 1e9f;
      return start + d / speed;
    }

    case ForceMode::AllAtStart:
    default:
      return start;
  }
}

/** @brief Initial linear velocity, px/frame, at the moment a shard detaches. */
SHP_HD inline Vec2 InitialVelocity(const ShardSim& s, const ForceParams& fp,
                                   const PhysicsParams& pp, const FrameScale& fs,
                                   uint32_t shardId, float mass) {
  Vec2 v(0.0f, 0.0f);

  const ForceField* fields[2] = {&fp.f1, &fp.f2};
  for (int i = 0; i < 2; ++i) {
    const ForceField& f = *fields[i];
    if (!f.enabled) continue;

    const float px = f.posX * fs.width;
    const float py = f.posY * fs.height;
    const float r = f.radius * fs.width;
    if (r <= 1e-6f) continue;

    const Vec2 away(s.cx - px, s.cy - py);
    const float dist = Length(away);
    const float k = ApplyFalloff(dist / r, fp.falloff);
    if (k <= 0.0f) continue;

    // A shard sitting exactly on the force point has no direction to go, so it
    // is given one from its own hash rather than left as the only piece that
    // never moves.
    Vec2 dir = Normalize(away);
    if (dir.x == 0.0f && dir.y == 0.0f) {
      const float a = Hash01(shardId, uint32_t(fp.seed), kChanVelAngle) * 2.0f * kPi;
      dir = Vec2(cosf(a), sinf(a));
    }

    v += dir * (f.strength * k * kVelocityPerStrength * fs.width);
  }

  // Randomness scatters both the direction and the magnitude. Scaling the jitter
  // by the speed the shard already has keeps a gentle shatter gentle -- an
  // absolute jitter would dominate whenever the forces were weak, and every
  // piece would fly off at the same speed regardless of the strength dial.
  if (pp.randomness > 0.0f) {
    const float speed = Length(v);
    const float a = Hash01(shardId, uint32_t(fp.seed), kChanVelAngle) * 2.0f * kPi;
    const float m = Hash01(shardId, uint32_t(fp.seed), kChanVelMag);
    v += Vec2(cosf(a), sinf(a)) * (speed * pp.randomness * m);
  }

  return v * (1.0f / mass);
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
