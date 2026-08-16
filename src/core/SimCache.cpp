#include "SimCache.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace shp {
namespace {

inline void mix(uint64_t& h, uint64_t v) {
  h = (h ^ v) * 1099511628211ull;
}

// Floats are folded through their bit pattern so that a change too small to see
// still counts as a change. Quantising first would be wrong here: unlike the
// content hash, where float noise in pixels should be ignored, a nudge to
// gravity really does mean a different simulation.
inline void mixF(uint64_t& h, float f) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(f), "float is not 32 bits");
  std::memcpy(&bits, &f, sizeof(bits));
  mix(h, uint64_t(bits));
}

inline void mixForce(uint64_t& h, const ForceField& f) {
  mix(h, f.enabled ? 1u : 0u);
  mix(h, uint64_t(f.shape));
  mixF(h, f.posX);
  mixF(h, f.posY);
  mixF(h, f.radius);
  mixF(h, f.direction);
  mixF(h, f.width);
  mixF(h, f.strength);
  mixF(h, f.delay);
}

}  // namespace

uint64_t HashSimParams(const ForceParams& fp, const PhysicsParams& pp,
                       const std::vector<Collider>& colliders) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a offset basis

  // Order matters and is preserved, so moving a collider changes the key even if
  // the set of positions happens to be the same afterwards.
  mix(h, uint64_t(colliders.size()));
  for (const Collider& c : colliders) {
    // The shape belongs here as much as the coordinates do. It is the one field
    // that is not a number on screen, which makes it the one a hash is likeliest
    // to be written without -- and the same two points mean something entirely
    // different as a box than as a line.
    mix(h, uint64_t(c.shape));
    mixF(h, c.x1);
    mixF(h, c.y1);
    mixF(h, c.x2);
    mixF(h, c.y2);
    mixF(h, c.angle);
  }

  mixForce(h, fp.f1);
  mixForce(h, fp.f2);
  mix(h, uint64_t(fp.mode));
  mixF(h, fp.waveSpeed);
  mix(h, uint64_t(fp.falloff));
  mix(h, uint64_t(int64_t(fp.startFrame)));
  mix(h, uint64_t(int64_t(fp.seed)));

  mixF(h, pp.rotationSpeed);
  mixF(h, pp.randomness);
  mixF(h, pp.viscosity);
  mixF(h, pp.massVariance);
  mixF(h, pp.gravity);
  mixF(h, pp.gravityDirection);
  mixF(h, pp.bounciness);
  mixF(h, pp.friction);
  mix(h, pp.collideShards ? 1u : 0u);
  mix(h, pp.floorEnabled ? 1u : 0u);
  mixF(h, pp.floorY);
  mix(h, pp.wallsEnabled ? 1u : 0u);
  mix(h, uint64_t(int64_t(pp.substeps)));

  // Deliberately NOT hashed: fade and shrink. They are applied after the
  // simulation, so changing them must not throw away a run.
  return h;
}

void SimCache::rebuild(const std::shared_ptr<const ShatterGeometry>& geom, const ForceParams& fp,
                       const PhysicsParams& pp, const FrameScale& fs,
                       const std::vector<Collider>& colliders) {
  _geomKeepAlive = geom;
  _fp = fp;
  _pp = pp;
  _fs = fs;
  _startFrame = fp.startFrame;

  _world.reset(geom->shards, fp, pp, fs, colliders);
  _frames.clear();
  _simulatedUpTo = -1;

  // Frame zero of the run is the state before any stepping: everything at rest.
  snapshot();
}

void SimCache::snapshot() {
  const std::vector<phys::Body>& bodies = _world.bodies();
  _frames.emplace_back();
  std::vector<ShardPose>& dst = _frames.back();
  dst.resize(bodies.size());

  for (std::size_t i = 0; i < bodies.size(); ++i) {
    const phys::Body& b = bodies[i];
    ShardPose& p = dst[i];
    p.x = b.pos.x;
    p.y = b.pos.y;
    p.angle = b.angle;
    p.vx = b.vel.x;
    p.vy = b.vel.y;
    p.w = b.w;
    p.scale = 1.0f;
    p.alpha = 1.0f;
    p.active = b.active;
  }
  ++_simulatedUpTo;
}

void SimCache::extendTo(int frame) {
  const int target = std::min(frame, kMaxSimFrames - 1);
  while (_simulatedUpTo < target) {
    // The frame being stepped INTO, which is what decides who detaches.
    _world.step(float(_startFrame + _simulatedUpTo + 1));
    snapshot();
  }
}

void SimCache::getPoses(const std::shared_ptr<const ShatterGeometry>& geom, const ForceParams& fp,
                        const PhysicsParams& pp, const RenderParams& rp, const FrameScale& fs,
                        const DepthParams& dp, const std::vector<Collider>& colliders, float t,
                        std::vector<ShardPose>& out) {
  if (!geom || geom->shards.empty()) {
    out.clear();
    return;
  }

  std::lock_guard<std::mutex> lock(_mutex);

  SimKey key;
  key.geom = geom.get();
  key.paramHash = HashSimParams(fp, pp, colliders);

  if (key != _key || _frames.empty()) {
    _key = key;
    rebuild(geom, fp, pp, fs, colliders);
  }

  // Before the shatter starts there is nothing to look up: every piece is at
  // rest, which is the all-zero pose.
  const float rel = t - float(_startFrame);
  if (rel <= 0.0f) {
    out.assign(geom->shards.size(), ShardPose{});
    return;
  }

  const int whole = int(std::floor(rel));
  extendTo(whole + 1);

  const int idx = std::min(whole, _simulatedUpTo);
  out = _frames[std::size_t(idx)];

  // Sub-frame interpolation by extrapolating along the stored velocities, the
  // same trick motion blur uses. Rendering a fractional time is rare -- Resolve
  // asks on whole frames -- but retimes and speed ramps do it, and snapping to
  // the frame below would make a slowed shatter visibly stepped.
  const float frac = rel - float(whole);
  if (frac > 1e-3f && idx == whole) {
    for (std::size_t i = 0; i < out.size(); ++i) {
      ShardPose& p = out[i];
      if (!p.active) continue;
      p.x += p.vx * frac;
      p.y += p.vy * frac;
      p.angle += p.w * frac;
    }
  }

  // Fade, shrink and depth are all applied here, after the simulation, so that
  // changing any of them never invalidates a run. Age is measured from the
  // shard's own detach frame, which the ballistic path computes the same way.
  const bool wantDepth = dp.spread > 0.0f || dp.bias != 0.0f || dp.tumble > 0.0f;
  if (rp.fadeDuration <= 0.0f && !wantDepth) return;

  const float drag = pp.viscosity * kDragScale;

  for (std::size_t i = 0; i < out.size(); ++i) {
    const Shard& sh = geom->shards[i];
    ShardSim ss;
    ss.cx = sh.cx;
    ss.cy = sh.cy;
    ss.area = sh.area;
    ss.radius = sh.radius;
    ss.inertiaPerMass = sh.inertiaPerMass;

    const float age = t - DetachFrame(ss, fp, fs);
    if (age < 0.0f) continue;

    if (rp.fadeDuration > 0.0f)
      ApplyFade(age, rp.fadeStart, rp.fadeDuration, rp.shrinkOnFade, out[i].alpha,
                out[i].scale);

    if (!wantDepth) continue;

    const uint32_t id = uint32_t(sh.id);
    const float mass = MassOf(ss, fs, pp, id, uint32_t(fp.seed));

    if (dp.spread > 0.0f || dp.bias != 0.0f) {
      // Launch speed taken from the impulses, matching the ballistic path
      // exactly -- the two modes must scatter through depth identically or
      // switching between them would look like a different effect.
      ForceImpulse imp[2];
      const int n = GatherImpulses(ss, fp, pp, fs, id, mass, imp);
      float launch = 0.0f;
      for (int e = 0; e < n; ++e) launch += Length(imp[e].linear);

      const float vz0 = InitialDepthVelocity(dp, launch, fs, id, uint32_t(fp.seed));
      Vec2 zp, zv;
      IntegrateDrag(Vec2(vz0, 0.0f), Vec2(0.0f, 0.0f), drag, age, zp, zv);
      out[i].z = zp.x;
      out[i].vz = zv.x;
    }

    if (dp.tumble > 0.0f) {
      float rate = 0.0f, axis = 0.0f;
      InitialTumble(dp, pp, id, uint32_t(fp.seed), rate, axis);
      float tAngle = 0.0f, tRate = 0.0f;
      IntegrateSpin(rate, drag, age, tAngle, tRate);
      out[i].tumble = tAngle;
      out[i].tumbleAxis = axis;
    }
  }
}

}  // namespace shp
