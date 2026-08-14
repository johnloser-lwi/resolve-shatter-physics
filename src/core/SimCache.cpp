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

uint64_t HashSimParams(const ForceParams& fp, const PhysicsParams& pp) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a offset basis

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
                       const PhysicsParams& pp, const FrameScale& fs) {
  _geomKeepAlive = geom;
  _fp = fp;
  _pp = pp;
  _fs = fs;
  _startFrame = fp.startFrame;

  _world.reset(geom->shards, fp, pp, fs);
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
                        float t, std::vector<ShardPose>& out) {
  if (!geom || geom->shards.empty()) {
    out.clear();
    return;
  }

  std::lock_guard<std::mutex> lock(_mutex);

  SimKey key;
  key.geom = geom.get();
  key.paramHash = HashSimParams(fp, pp);

  if (key != _key || _frames.empty()) {
    _key = key;
    rebuild(geom, fp, pp, fs);
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

  // Fade and shrink are applied here, after the simulation, so that changing
  // them never invalidates a run. Age is measured from the shard's own detach
  // frame, which the ballistic path computes the same way.
  if (rp.fadeDuration > 0.0f) {
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
      ApplyFade(age, rp.fadeStart, rp.fadeDuration, rp.shrinkOnFade, out[i].alpha,
                out[i].scale);
    }
  }
}

}  // namespace shp
