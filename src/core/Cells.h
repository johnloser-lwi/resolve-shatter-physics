// Deciding where the image breaks.
//
// Five patterns, one output contract: an int32 label per pixel plus a Shard per
// label. The label image doubles as the render stencil, which is what lets the
// compositor draw an arbitrarily shaped piece with a single integer compare
// instead of carrying a mask per shard.
//
// Everything here is authored in NORMALIZED [0,1] space and scaled to pixels
// exactly once, at the end. That is not tidiness -- it is what makes a proxy
// render break into the same pieces as a full-resolution one. Text Animator
// learned this the hard way: a control denominated in pixels quietly means
// something different at every resolution, so a shatter tuned in a half-res
// timeline would rearrange itself on delivery.
#pragma once

#include "ImageView.h"
#include "ShardGeom.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace shp {

enum class Pattern {
  Voronoi = 0,   ///< irregular glass shards
  Bricks  = 1,
  Grid    = 2,
  Alpha   = 3,   ///< one connected alpha region per piece
  Map     = 4,   ///< regions of constant colour in a second input
};

struct ShapeParams {
  Pattern pattern = Pattern::Voronoi;

  int cellCount = 50;         ///< target number of cells before alpha clipping
  float cellJitter = 0.75f;   ///< 0 = a regular lattice, 1 = fully scattered seeds
  int brickRows = 8;
  float brickOffset = 0.5f;   ///< row-to-row stagger, in cell widths
  int mapLevels = 16;         ///< posterisation levels per channel for Pattern::Map

  float alphaThreshold = 0.15f;
  float minPieceArea = 0.0002f;  ///< fraction of the frame's area
  float bridgeRadius = 0.0f;     ///< fraction of frame height
  int shapeSeed = 1;

  /// Whether the caller needs collision polygons. Building them costs real time
  /// on a thousand-shard frame and the ballistic mode never looks at them.
  bool buildPolygons = false;

  bool operator==(const ShapeParams& o) const {
    return pattern == o.pattern && cellCount == o.cellCount && cellJitter == o.cellJitter &&
           brickRows == o.brickRows && brickOffset == o.brickOffset && mapLevels == o.mapLevels &&
           alphaThreshold == o.alphaThreshold && minPieceArea == o.minPieceArea &&
           bridgeRadius == o.bridgeRadius && shapeSeed == o.shapeSeed &&
           buildPolygons == o.buildPolygons;
  }
  bool operator!=(const ShapeParams& o) const { return !(*this == o); }
};

/** @brief Break @p src into pieces.
 *
 * @param map optional second input, used only by Pattern::Map. When the pattern
 *            asks for a map and none is connected, the call falls back to
 *            Voronoi rather than returning nothing -- a missing input should
 *            look like a default, not like a broken effect.
 *
 * Never returns null. An empty frame yields a geometry with no shards, which the
 * compositor renders as a transparent frame.
 */
std::shared_ptr<const ShatterGeometry> buildGeometry(const ConstImageView& src,
                                                     const ConstImageView* map,
                                                     const ShapeParams& params);

// ------------------------------------------------------------------ internals
// Exposed for the tests, which check the cell maps directly rather than only
// through a rendered frame.

/** @brief Per-pixel cell id for a generated pattern. Ids are dense from 1. */
std::vector<int32_t> buildCellMap(int w, int h, const ShapeParams& params, int& cellCountOut);

/** @brief Exact convex polygon of one Voronoi cell, by half-plane clipping.
 *
 * Marching the raster boundary would give a staircase with hundreds of vertices
 * that still has to be simplified; clipping gives the true polygon directly, and
 * it is convex by construction, which the solver requires. */
std::vector<Vec2> voronoiCellPolygon(int cellIndex, int w, int h, const ShapeParams& params);

}  // namespace shp
