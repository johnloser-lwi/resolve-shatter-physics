// Drawing the posed shards.
//
// Gather, not scatter: every destination pixel asks "which shards land on me",
// rather than every shard writing where it lands. Scatter is the shorter code on
// a CPU, but it makes the write order depend on the shard loop, which means it
// cannot be parallelised without either locks or a per-shard pass over the
// frame -- and it has no analogue at all in a CUDA kernel, where a thread owns
// one output pixel and nothing else. Writing it as a gather once means the GPU
// path in M6 is the same algorithm rather than a second implementation to keep
// in step.
//
// The "which shards" question is answered by binning transformed shard bounds
// into a coarse tile grid, so a pixel tests a handful of candidates instead of
// all thousand.
#pragma once

#include "Ballistic.h"
#include "ImageView.h"
#include "Sampler.h"
#include "ShardGeom.h"

#include <cstdint>
#include <vector>

namespace shp {

/// Tile edge in pixels. 32 keeps the CSR arrays small on a UHD frame while still
/// rejecting most shards for most pixels.
constexpr int kTileSize = 32;

/** @brief Which shards touch which tile, in compressed sparse row form.
 *
 * Built on the CPU in both paths: it is O(shards), not O(pixels), so moving it
 * to the GPU would save nothing and cost a kernel launch. */
struct TileGrid {
  int tilesX = 0, tilesY = 0;
  std::vector<int32_t> start;    ///< size tilesX*tilesY + 1
  std::vector<int32_t> shardIdx; ///< concatenated shard indices, ascending per tile

  int tileCount() const { return tilesX * tilesY; }
};

/// Upper bound on shutter samples. Eight is where the improvement stops being
/// visible on a piece moving fast enough to need blur at all.
constexpr int kMaxBlurSamples = 8;

/** @brief The per-shard transform and shading data the inner loop needs.
 *
 * Precomputed once per frame so the pixel loop does no trigonometry. */
struct ShardDraw {
  /// Destination pixel -> source pixel, one per shutter sample. With motion blur
  /// off only [0] is filled and count is 1, which is the same code path.
  Mat3 inv[kMaxBlurSamples];
  int count = 1;
  int32_t label = 0;            ///< what to expect in the label image
  float alpha = 1.0f;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0;  ///< destination bounds, clipped to the frame
};

struct BlurParams {
  bool enabled = false;
  float shutterAngle = 180.0f;  ///< degrees; 180 means the shutter is open half a frame
  int samples = 8;
};

struct CompositeParams {
  bool edgeAntialias = true;
  bool showPieces = false;  ///< diagnostic tint instead of the source pixels
};

/** @brief Build the draw list for one frame.
 *
 * Shards whose pose is at rest still get an entry: they have to be drawn where
 * they started, and giving them the identity transform means the inner loop has
 * no special case for them.
 *
 * Motion blur is built here rather than in the simulation because the sub-frame
 * poses come from extrapolating the stored velocities, which both simulation
 * modes already provide. No extra frames are simulated and no neighbouring
 * source frames are fetched -- which is why this plugin can leave temporal clip
 * access switched off. */
void BuildDrawList(const ShatterGeometry& geom, const std::vector<ShardPose>& poses,
                   const BlurParams& blur, std::vector<ShardDraw>& out);

/** @brief Bin the draw list into tiles. */
void BuildTileGrid(const std::vector<ShardDraw>& draws, int width, int height, TileGrid& out);

/** @brief Composite the shards into @p dst.
 *
 * @param dst  cleared to transparent by this call; callers need not pre-clear.
 * @param src  the untransformed source frame, sampled through each shard's stencil.
 */
void Composite(const ImageView& dst, const ConstImageView& src, const ShatterGeometry& geom,
               const std::vector<ShardDraw>& draws, const TileGrid& tiles,
               const CompositeParams& params);

// ---------------------------------------------------------------- shared core
//
// Everything below is device-safe and takes raw pointers, because the CUDA
// kernel calls these exact functions rather than reimplementing them. That is
// the rule that keeps the two paths from drifting: CudaKernel.cu contains no
// maths of its own, and the parity test exists to prove it.

/** @brief Does the source position @p sx,@p sy belong to shard @p label?
 *
 * Nearest lookup, never bilinear: the label image holds ids, and interpolating
 * between id 3 and id 7 would give 5, which is a different shard entirely. */
SHP_HD inline bool StencilHit(const int32_t* labels, int lw, int lh, float sx, float sy,
                              int32_t label) {
  // The float tests come first: int() truncates toward zero, so -0.4 would
  // become 0 and a pixel just off the left edge would read the leftmost column.
  if (sx < 0.0f || sy < 0.0f) return false;
  const int ix = int(sx);
  const int iy = int(sy);
  if (ix >= lw || iy >= lh) return false;
  return labels[(long long)(iy) * lw + ix] == label;
}

/** @brief Shade one destination pixel from the shards binned to its tile.
 *
 * @param lo,hi the CSR range for this pixel's tile.
 * @param out   four premultiplied floats; written unconditionally.
 */
SHP_HD inline void ShadePixel(const int32_t* labels, int lw, int lh, const SampleView& src,
                              const ShardDraw* draws, const int32_t* tileShardIdx, int32_t lo,
                              int32_t hi, int x, int y, bool edgeAntialias, bool showPieces,
                              float* out) {
  out[0] = out[1] = out[2] = out[3] = 0.0f;

  const float dx = float(x) + 0.5f;
  const float dy = float(y) + 0.5f;

  for (int32_t k = lo; k < hi; ++k) {
    const ShardDraw& d = draws[tileShardIdx[k]];
    if (x < d.x1 || x >= d.x2 || y < d.y1 || y >= d.y2) continue;

    // Shutter samples ACCUMULATE premultiplied colour and are divided by the
    // sample count at the end -- they are not composited over one another.
    // Over-compositing N samples of alpha a/N gives 1-(1-a/N)^N, so a piece
    // standing still would come out at about 63% opacity instead of solid.
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool any = false;

    for (int s = 0; s < d.count; ++s) {
      float sx, sy;
      d.inv[s].Apply(dx, dy, sx, sy);

      float cov = 0.0f;
      if (edgeAntialias) {
        // Four stencil taps on a half-pixel diagonal. The shard's silhouette is
        // a hard raster edge, and a single tap leaves it visibly jagged once the
        // piece rotates off axis -- most obvious on the long straight cuts a
        // brick pattern produces.
        const float off[4][2] = {
            {-0.25f, -0.25f}, {0.25f, -0.25f}, {-0.25f, 0.25f}, {0.25f, 0.25f}};
        for (int q = 0; q < 4; ++q) {
          float ax, ay;
          d.inv[s].Apply(dx + off[q][0], dy + off[q][1], ax, ay);
          if (StencilHit(labels, lw, lh, ax, ay, d.label)) cov += 0.25f;
        }
      } else {
        cov = StencilHit(labels, lw, lh, sx, sy, d.label) ? 1.0f : 0.0f;
      }
      if (cov <= 0.0f) continue;

      float px[4];
      if (showPieces) {
        const uint32_t id = uint32_t(d.label);
        px[0] = 0.25f + 0.75f * Hash01(id, 11u);
        px[1] = 0.25f + 0.75f * Hash01(id, 22u);
        px[2] = 0.25f + 0.75f * Hash01(id, 33u);
        px[3] = 1.0f;
      } else {
        SampleBilinear(src, sx, sy, kEdgeBlack, px);
      }

      // Coverage and the shard's own fade both scale all four channels: the
      // source is premultiplied, so fading means scaling the premultiplied
      // colour too. Touching alpha alone leaves the colour too bright and the
      // fade looks wrong over anything but black.
      const float a = cov * d.alpha;
      acc[0] += px[0] * a;
      acc[1] += px[1] * a;
      acc[2] += px[2] * a;
      acc[3] += px[3] * a;
      any = true;
    }

    if (!any) continue;

    const float inv = 1.0f / float(d.count);
    acc[0] *= inv;
    acc[1] *= inv;
    acc[2] *= inv;
    acc[3] *= inv;
    CompositeOver(out, acc);
  }
}

}  // namespace shp
