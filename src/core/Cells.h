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
  Manual  = 5,   ///< no generated pattern at all; only the slices you draw
};

/** @brief A cut drawn by hand, in normalized [0,1] frame coordinates.
 *
 * A finite segment, not an infinite line, and that distinction is the whole
 * design. An infinite line would also cut every other object that happened to
 * lie along it, which is astonishing when you only meant to halve one letter. A
 * segment cuts what it actually crosses and nothing else -- and a segment that
 * stops halfway through a piece does not separate it at all, which is exactly
 * how a real knife behaves.
 */
struct Slice {
  float x1 = 0.0f, y1 = 0.0f;
  float x2 = 0.0f, y2 = 0.0f;

  bool operator==(const Slice& o) const {
    return x1 == o.x1 && y1 == o.y1 && x2 == o.x2 && y2 == o.y2;
  }
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

  /// Cuts drawn by hand. These apply on TOP of whatever pattern is selected, so
  /// a generated shatter can be tidied up by hand rather than being an
  /// all-or-nothing choice. Pattern::Manual simply generates nothing, leaving
  /// these as the only thing that separates anything.
  std::vector<Slice> slices;

  /// Whether the caller needs collision polygons. Building them costs real time
  /// on a thousand-shard frame and the ballistic mode never looks at them.
  bool buildPolygons = false;

  /// Whether the bevel needs its distance-to-edge field. Two more bytes per
  /// pixel and a pair of sweeps over the frame, so it is not built for a flat
  /// shatter that will never read it.
  bool buildEdgeField = false;

  bool operator==(const ShapeParams& o) const {
    return pattern == o.pattern && cellCount == o.cellCount && cellJitter == o.cellJitter &&
           brickRows == o.brickRows && brickOffset == o.brickOffset && mapLevels == o.mapLevels &&
           alphaThreshold == o.alphaThreshold && minPieceArea == o.minPieceArea &&
           bridgeRadius == o.bridgeRadius && shapeSeed == o.shapeSeed &&
           buildPolygons == o.buildPolygons && buildEdgeField == o.buildEdgeField &&
           slices == o.slices;
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
