// Making a stepped simulation survive a host that renders frames in any order.
//
// This is the whole reason the rigid-body mode is possible at all. A solver only
// knows how to go forward one frame at a time, but Resolve will ask for frame
// 200, then frame 3, then frame 201, and will keep whichever of them it likes in
// its own cache. Answering "simulate from the start each time" would be
// unusable, and answering "step from wherever we happen to be" would give a
// different picture depending on the order the user scrubbed in.
//
// So: simulate forward once, remember every frame's poses, and serve them.
// Because the run is deterministic (see Physics2D.h), a rebuild after a
// parameter change reproduces exactly the same trajectory, which is what makes
// scrubbing backward show the same thing scrubbing forward did.
//
// Storage is a few dozen bytes per shard per frame -- a thousand shards over a
// thousand frames is about 40 MB, and the cap below keeps the worst case bounded.
#pragma once

#include "Ballistic.h"
#include "Forces.h"
#include "Physics2D.h"
#include "ShardGeom.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace shp {

/// Beyond this many simulated frames the cache stops growing and keeps serving
/// the last one. Anything still on screen after four thousand frames of gravity
/// is not a shatter any more.
constexpr int kMaxSimFrames = 4096;

/** @brief Everything that changes a trajectory, folded into one comparable key.
 *
 * The geometry is keyed by POINTER, not by content: the analysis cache already
 * guarantees one ShatterGeometry object per (content, shape params), so two
 * geometries that are equal are the same object. Holding the shared_ptr
 * alongside is what keeps that address from being recycled underneath us. */
struct SimKey {
  const ShatterGeometry* geom = nullptr;
  uint64_t paramHash = 0;

  bool operator==(const SimKey& o) const {
    return geom == o.geom && paramHash == o.paramHash;
  }
  bool operator!=(const SimKey& o) const { return !(*this == o); }
};

/** @brief FNV-1a over the force and physics settings.
 *
 * Hashing the raw bytes of the structs would be shorter and wrong: padding is
 * uninitialised, so two identical settings could hash differently and quietly
 * force a rebuild on every frame. */
uint64_t HashSimParams(const ForceParams& fp, const PhysicsParams& pp,
                       const std::vector<Collider>& colliders);

class SimCache {
 public:
  /** @brief Poses at clip time @p t, simulating forward as far as needed.
   *
   * Thread-safe: the effect is eRenderInstanceSafe, so two host threads may ask
   * at once. The lock is held across the simulation, which means a backward
   * scrub after a parameter change makes one thread wait for the rebuild rather
   * than starting a second one.
   */
  void getPoses(const std::shared_ptr<const ShatterGeometry>& geom, const ForceParams& fp,
                const PhysicsParams& pp, const RenderParams& rp, const FrameScale& fs,
                const std::vector<Collider>& colliders, float t, std::vector<ShardPose>& out);

 private:
  void rebuild(const std::shared_ptr<const ShatterGeometry>& geom, const ForceParams& fp,
               const PhysicsParams& pp, const FrameScale& fs,
               const std::vector<Collider>& colliders);
  void extendTo(int frame);
  void snapshot();

  std::mutex _mutex;
  SimKey _key;
  std::shared_ptr<const ShatterGeometry> _geomKeepAlive;

  phys::World _world;
  ForceParams _fp;
  PhysicsParams _pp;
  FrameScale _fs;

  /// _frames[i] holds the poses at clip frame _startFrame + i.
  std::vector<std::vector<ShardPose>> _frames;
  int _startFrame = 0;
  int _simulatedUpTo = -1;  ///< index of the last entry in _frames
};

}  // namespace shp
