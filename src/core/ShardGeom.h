// One piece of a shattered image, and the collection of them.
//
// A shard is defined twice over, deliberately:
//
//   - As pixels, by its id in ShatterGeometry::labelImage. This is what the
//     compositor gates on, and it is exact: a shard can be any shape at all,
//     including one with holes, and it costs nothing to test because the label
//     image is one int32 read.
//
//   - As a convex polygon, in Shard::poly. This is a *proxy* used only by the
//     rigid-body solver, which needs edges and normals to find contacts. It is
//     convex even when the pixels are not, because convex-convex contact is the
//     one case a small solver can do correctly.
//
// Keeping both is what lets a piece look like a torn letter while colliding like
// something the physics can actually reason about. Mass and inertia come from
// the true pixel area rather than the hull, so a hollow ring does not get heavier
// just because its hull is solid.
#pragma once

#include "ImageView.h"
#include "TransformMath.h"

#include <cstdint>
#include <vector>

namespace shp {

struct Shard {
  int id = 0;             ///< the value this shard has in labelImage; 1-based, never 0
  float cx = 0, cy = 0;   ///< pixel centroid, the pivot everything rotates about
  float area = 0;         ///< true pixel count, the mass source
  RectI bbox;             ///< tight pixel bounds of the shard at rest

  /// Convex outline in absolute pixels, counter-clockwise in the top-down space
  /// core code uses. Empty when physics is not going to be run on this geometry.
  std::vector<Vec2> poly;

  /// Second moment of the polygon about its centroid, per unit mass. Purely
  /// geometric, so it belongs here; the actual rotational inertia is this times
  /// whatever mass the simulation decides on.
  ///
  /// Mass itself deliberately does NOT live here. It depends on massVariance,
  /// which is a physics control, and baking it into the geometry would make
  /// nudging the mass jitter re-run the whole segmentation.
  float inertiaPerMass = 0;

  /// Distance from the centroid to the furthest polygon vertex. The broadphase
  /// uses it as a bounding radius, and the anti-tunnelling clamp uses the
  /// smallest one in the scene.
  float radius = 0;
};

// The result of shattering: which pixel belongs to which piece, plus the pieces.
//
// Immutable once built, held by shared_ptr<const>, and its POINTER IDENTITY is
// what keys every downstream cache -- the auto-mask rule. That works because the
// analysis cache guarantees one object per (content, shape params), so two
// geometries that are equal are also the same object, and holding the shared_ptr
// keeps the address from being recycled under a cache that is still using it.
struct ShatterGeometry {
  int width = 0, height = 0;

  /// One int32 per pixel: 0 is background, n > 0 means shards[n-1].
  std::vector<int32_t> labelImage;
  std::vector<Shard> shards;

  /// Averages over the shards, computed once at build time.
  ///
  /// The force and spin scales are expressed relative to "an average piece", so
  /// that raising the piece count makes a shatter finer rather than making every
  /// piece fly further. Recomputing these per frame would be cheap but would put
  /// an O(shards) loop in the render path for a number that cannot change
  /// without rebuilding the geometry anyway.
  float meanArea = 1.0f;
  float meanRadius = 1.0f;

  int32_t labelAt(int x, int y) const {
    return labelImage[std::size_t(y) * std::size_t(width) + std::size_t(x)];
  }

  bool empty() const { return shards.empty(); }
};

// Where a shard is at some instant. Both simulation modes produce these, which
// is what lets the compositor stay ignorant of which mode ran.
//
// The translation is a DELTA from the shard's rest position, not an absolute
// position. That way an untouched shard is all-zeros and composites back exactly
// where it started, with no dependence on its centroid having been computed to
// the same last bit.
struct ShardPose {
  float x = 0, y = 0;      ///< displacement of the centroid, pixels
  float angle = 0;         ///< rotation about the centroid, radians
  float vx = 0, vy = 0;    ///< velocity, px/frame -- motion blur reads these
  float w = 0;             ///< angular velocity, rad/frame
  float scale = 1;         ///< uniform, about the centroid
  float alpha = 1;         ///< multiplied into the sampled pixel
  bool active = false;     ///< false before this shard's detach frame

  bool isRest() const {
    return !active && x == 0 && y == 0 && angle == 0 && scale == 1 && alpha == 1;
  }
};

// ------------------------------------------------------------------ geometry

/** @brief Convex hull of a point set, counter-clockwise, via Andrew's monotone chain.
 *
 * Returns fewer than 3 points only for degenerate input (all points collinear or
 * coincident), which callers must treat as "no physics proxy" rather than as a
 * polygon -- a zero-area hull has no normals and would divide by zero in the
 * inertia calculation. */
std::vector<Vec2> ConvexHull(std::vector<Vec2> pts);

/** @brief Signed area of a polygon. Positive when counter-clockwise. */
float PolygonArea(const std::vector<Vec2>& poly);

/** @brief Area centroid of a polygon. */
Vec2 PolygonCentroid(const std::vector<Vec2>& poly);

/** @brief Second moment of area about the polygon's own centroid, per unit mass.
 *
 * Returns the value that, multiplied by mass, gives the rotational inertia. Kept
 * per-unit-mass so that mass jitter does not require recomputing the integral. */
float PolygonInertiaPerMass(const std::vector<Vec2>& poly, const Vec2& centroid);

/** @brief Clip a convex polygon by the half-plane {p : dot(n, p) <= d}.
 *
 * Sutherland-Hodgman with a single plane. This is how Voronoi cells are built:
 * start from a rectangle and clip once per neighbouring seed. */
std::vector<Vec2> ClipHalfPlane(const std::vector<Vec2>& poly, const Vec2& n, float d);

}  // namespace shp
