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

  float z = 0.0f;      ///< only used to sort; the kernel never reads it
  float shade = 1.0f;  ///< aerial perspective, multiplied into the sampled pixel
  float angle = 0.0f;  ///< the piece's rotation, for turning the bevel normals
};

/** @brief Where the destination sits relative to the source.
 *
 * On Resolve's Edit page these are the same buffer size and this is all zeros.
 * Inside a Fusion comp they are not: Fusion crops an input to its domain of
 * definition, so the source can be smaller than the frame being written and
 * offset within it. Everything in core works in SOURCE space -- the label image,
 * the shard centroids, the edge field -- so the destination has to be mapped
 * into that space rather than assumed to coincide with it.
 *
 * Getting this wrong in Fusion does not look like an offset. The source view is
 * built from the destination's bounds, `getPixelAddress` returns null for a
 * pixel the source does not have, and the frame comes out black.
 */
struct FrameLayout {
  int dstWidth = 0, dstHeight = 0;  ///< 0 means "same as the source"
  int offX = 0, offY = 0;           ///< destination (0,0) is source (offX, offY)
};

struct BlurParams {
  bool enabled = false;
  float shutterAngle = 180.0f;  ///< degrees; 180 means the shutter is open half a frame
  int samples = 8;
};

struct CompositeParams {
  bool edgeAntialias = true;
  bool showPieces = false;  ///< diagnostic tint instead of the source pixels

  /// Inner bevel. 0 switches it off, and the edge field is then not even built.
  float bevel = 0.0f;
  float bevelWidth = 8.0f;   ///< pixels
  float lightX = -0.707f;    ///< unit vector, the direction light travels
  float lightY = -0.707f;
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
                   const BlurParams& blur, const DepthParams& depth, const FrameScale& fs,
                   std::vector<ShardDraw>& out, const FrameLayout& layout = FrameLayout{});

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

/** @brief Brightness at one point inside a piece, from its bevel.
 *
 * Returns a multiplier: above 1 on the edge facing the light, below 1 on the
 * edge facing away, and exactly 1 everywhere further in than the bevel's width.
 *
 * @param edgeDist,edgeAngle the packed field, or null when there is no bevel
 * @param shardAngle         the piece's current rotation, added to the stored
 *                           normal so the lighting stays fixed in the world
 *                           while the piece turns under it
 */
SHP_HD inline float BevelShade(const uint8_t* edgeDist, const uint8_t* edgeAngle, int lw,
                               int lh, int ix, int iy, float shardAngle, float strength,
                               float widthPx, float lightX, float lightY) {
  if (!edgeDist || strength <= 0.0f || widthPx <= 0.01f) return 1.0f;

  const long long p = (long long)(iy) * lw + ix;
  const float d = float(edgeDist[p]);
  if (d >= widthPx) return 1.0f;   // deep inside: flat

  // Falloff across the bevel. Squared so the band hugs the edge rather than
  // washing across the whole piece, which is what a chamfer actually looks like.
  const float t = 1.0f - d / widthPx;
  const float falloff = t * t;

  // The stored normal points into the piece, in the piece's own frame. Adding
  // the piece's rotation puts it back into the world, which is the difference
  // between a light that stays put and one that spins with the debris.
  const float a = float(edgeAngle[p]) * (2.0f * kPi / 255.0f) + shardAngle;
  const float nx = cosf(a);
  const float ny = sinf(a);

  // Lambert against the surface of the chamfer. The outward face is the
  // negation of the inward normal, so this dot comes out positive exactly where
  // the edge is turned toward the light.
  const float lambert = nx * lightX + ny * lightY;

  // Highlights are weaker than shadows on purpose: a bevel catching light reads
  // as a thin bright line, while the shadow side is a broader, softer mass.
  const float amount = lambert > 0.0f ? lambert * 0.6f : lambert;
  return Clampf(1.0f + strength * falloff * amount, 0.0f, 4.0f);
}

/** @brief Shade one destination pixel from the shards binned to its tile.
 *
 * @param lo,hi the CSR range for this pixel's tile.
 * @param out   four premultiplied floats; written unconditionally.
 */
SHP_HD inline void ShadePixel(const int32_t* labels, const uint8_t* edgeDist,
                              const uint8_t* edgeAngle, int lw, int lh, const SampleView& src,
                              const ShardDraw* draws, const int32_t* tileShardIdx, int32_t lo,
                              int32_t hi, int x, int y, bool edgeAntialias, bool showPieces,
                              float bevel, float bevelWidth, float lightX, float lightY,
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

      // Bevel and aerial perspective touch COLOUR only, never alpha. Dimming a
      // piece's alpha would make it translucent rather than dark, and the edge
      // it is supposed to be shading would show whatever is behind it.
      float lum = d.shade;
      if (bevel > 0.0f) {
        const int bx = int(sx), by = int(sy);
        if (sx >= 0.0f && sy >= 0.0f && bx < lw && by < lh)
          lum *= BevelShade(edgeDist, edgeAngle, lw, lh, bx, by, d.angle, bevel, bevelWidth,
                            lightX, lightY);
      }

      // Coverage and the shard's own fade scale all four channels: the source is
      // premultiplied, so fading means scaling the premultiplied colour too.
      // Touching alpha alone leaves the colour too bright and the fade looks
      // wrong over anything but black.
      const float a = cov * d.alpha;
      acc[0] += px[0] * a * lum;
      acc[1] += px[1] * a * lum;
      acc[2] += px[2] * a * lum;
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
