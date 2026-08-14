// A small 2D rigid-body solver, written for exactly this job.
//
// Why not Box2D: what a shatter needs is convex polygons, gravity, one impulse
// each at launch, a floor, and plausible tumbling. No joints, no motors, no
// sleeping islands, no stacking that has to stay stable for a minute. Vendoring
// a general engine would mean auditing its allocator and threading model for
// determinism and static-CRT compatibility, to use perhaps eight hundred lines
// of it.
//
// Determinism is the hard requirement, not a nicety. SimCache replays this from
// the start frame whenever its key changes, and if two runs of the same inputs
// diverged, scrubbing backward would show a different shatter than scrubbing
// forward. That is guaranteed here by construction: single-threaded, fixed
// iteration order, no containers whose traversal order depends on addresses, and
// no reliance on anything the compiler is free to reassociate across runs.
//
// The solver is sequential impulses -- the same approach Box2D uses -- with
// speculative-free discrete collision. There is no continuous collision
// detection: a piece moving further than its own radius in one substep can pass
// through a thin obstacle. The substep count is exposed as "Accuracy" for that
// reason, and the integrator clamps per-substep travel as a backstop.
#pragma once

#include "Forces.h"
#include "ShardGeom.h"

#include <vector>

namespace shp {

enum class ColliderShape {
  Line   = 0,  ///< a bare segment: a ramp, a shelf, a wall at an angle
  Rect   = 1,  ///< axis-aligned box, solid throughout
  Circle = 2,  ///< solid disc
};

/** @brief A static obstacle drawn by hand, in normalized [0,1] coordinates.
 *
 * Every shape is stored as exactly two control points, which is what lets one
 * pair of drag handles edit all three:
 *
 *   Line    p1 and p2 are the ends
 *   Rect    p1 and p2 are opposite corners
 *   Circle  p1 is the centre, p2 is a point on the rim
 *
 * A rect is a real convex body rather than four separate lines, so a piece
 * cannot end up inside it -- four lines leave an interior that anything already
 * within simply stays in.
 *
 * Deliberately NOT part of PhysicsParams. That struct is passed by reference to
 * functions marked SHP_HD, and giving it a std::vector member would make a
 * device-compiled header depend on host-only types.
 */
struct Collider {
  ColliderShape shape = ColliderShape::Rect;
  float x1 = 0.0f, y1 = 0.0f;
  float x2 = 0.0f, y2 = 0.0f;

  bool operator==(const Collider& o) const {
    return shape == o.shape && x1 == o.x1 && y1 == o.y1 && x2 == o.x2 && y2 == o.y2;
  }
};

namespace phys {

struct Body {
  // State.
  Vec2 pos{0.0f, 0.0f};    ///< displacement of the centroid from rest, pixels
  float angle = 0.0f;      ///< radians about the centroid
  Vec2 vel{0.0f, 0.0f};    ///< px/frame
  float w = 0.0f;          ///< rad/frame

  // Constants for this shard.
  Vec2 rest{0.0f, 0.0f};   ///< centroid at rest, absolute pixels
  float radius = 0.0f;

  /// Effective inverse mass: zero while the piece has not detached yet, and
  /// again once it is culled. Zero means immovable, which is exactly the
  /// behaviour wanted in both cases.
  float invMass = 0.0f;
  float invInertia = 0.0f;

  /// What invMass/invInertia become at detach. Kept separately so that
  /// activation is a copy rather than a recomputation -- the mass depends on a
  /// hash draw, and drawing it twice would be one more chance to disagree.
  float invMassFull = 0.0f;
  float invInertiaFull = 0.0f;

  /// Vertices relative to the centroid, at rest. The solver rotates these into
  /// world space per substep rather than storing a transformed copy.
  std::vector<Vec2> local;

  float detachFrame = 0.0f;
  bool active = false;     ///< false until detachFrame; an inactive body is an obstacle
  bool culled = false;     ///< left the frame for good and stopped being simulated

  /// Forces waiting to arrive, in time order. A delayed force is a genuine
  /// second shove mid-flight, so it cannot be folded into the launch velocity.
  ForceImpulse pending[2];
  int pendingCount = 0;
  int pendingNext = 0;     ///< how many have already been applied
  float spinAtDetach = 0.0f;

  Vec2 worldPos() const { return rest + pos; }
};

struct WorldParams {
  Vec2 gravity{0.0f, 0.0f};
  float drag = 0.0f;
  float restitution = 0.3f;
  float friction = 0.4f;
  bool collideShards = true;
  bool floorEnabled = true;
  float floorY = 1080.0f;   ///< absolute pixels, top-down
  bool wallsEnabled = false;
  float leftX = 0.0f;
  float rightX = 1920.0f;
  int substeps = 4;

  /// How far outside the frame a body may drift before it stops being simulated.
  float cullMargin = 2000.0f;
  float frameW = 1920.0f;
  float frameH = 1080.0f;
};

class World {
 public:
  void reset(const std::vector<Shard>& shards, const ForceParams& fp, const PhysicsParams& pp,
             const FrameScale& fs, const std::vector<Collider>& colliders);

  /// Advance exactly one frame, in `substeps` equal pieces.
  /// @param frame the frame being stepped INTO. Bodies whose detach time has
  ///        arrived are released, and impulses that fall due within the frame
  ///        are applied at substep resolution.
  void step(float frame);

  const std::vector<Body>& bodies() const { return _bodies; }
  const WorldParams& params() const { return _params; }

  /// Where the simulation currently sits, in absolute clip frames. Advances by
  /// one substep at a time, which is the resolution impulses land at.
  float time() const { return _time; }

 private:
  void integrate(float dt);
  void releaseDue(float now);
  void solveContacts(float dt);
  void collectShardContacts();
  void collectStaticContacts();
  void collectColliderContacts();

  /// Convex-vs-convex contacts, appended to _contacts. @p ib may be -1 for a
  /// static obstacle. Shared by shard-shard and shard-rectangle, so both get the
  /// same reference-face clipping rather than one of them getting an
  /// approximation nobody remembers is there.
  void addPolyContacts(int ia, int ib, const std::vector<Vec2>& hullA,
                       const std::vector<Vec2>& hullB);

  struct Contact {
    int a = -1;          ///< body index; b < 0 means a static plane
    int b = -1;
    Vec2 normal{0.0f, 0.0f};  ///< unit, pointing from a toward b
    Vec2 point{0.0f, 0.0f};
    float depth = 0.0f;
    float normalImpulse = 0.0f;
    float tangentImpulse = 0.0f;
  };

  std::vector<Body> _bodies;
  std::vector<Contact> _contacts;
  std::vector<Collider> _colliders;   ///< absolute pixels, resolved at reset
  WorldParams _params;
  float _time = 0.0f;

  // Broadphase scratch, kept between steps so a long simulation is not a long
  // series of allocations.
  std::vector<int> _cellStart;
  std::vector<int> _cellItems;
  std::vector<Vec2> _worldVerts;  ///< scratch for one body's transformed hull
  float _cellSize = 64.0f;
  int _gridW = 1, _gridH = 1;
  float _gridMinX = 0.0f, _gridMinY = 0.0f;
};

}  // namespace phys
}  // namespace shp
