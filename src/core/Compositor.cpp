#include "Compositor.h"

#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace shp {

void BuildDrawList(const ShatterGeometry& geom, const std::vector<ShardPose>& poses,
                   const BlurParams& blur, const DepthParams& depth, const FrameScale& fs,
                   std::vector<ShardDraw>& out) {
  out.clear();
  out.reserve(geom.shards.size());

  const int nSamples =
      blur.enabled ? std::max(1, std::min(kMaxBlurSamples, blur.samples)) : 1;
  const float shutter = blur.enabled ? std::max(0.0f, blur.shutterAngle) / 360.0f : 0.0f;

  for (std::size_t i = 0; i < geom.shards.size(); ++i) {
    const Shard& sh = geom.shards[i];
    const ShardPose& p = i < poses.size() ? poses[i] : ShardPose{};

    if (p.alpha <= 0.0f) continue;  // fully faded: nothing to draw

    ShardDraw d;
    d.label = sh.id;
    d.alpha = p.alpha;
    d.z = p.z;
    d.angle = p.angle;
    d.shade = AerialShade(p.z, depth, fs);

    // A resting piece gets one sample however the blur is set: extrapolating a
    // zero velocity would produce identical samples at real cost, and a piece
    // that has not detached must stay perfectly crisp.
    const bool moving = p.active && (p.vx != 0.0f || p.vy != 0.0f || p.w != 0.0f);
    d.count = (blur.enabled && moving) ? nSamples : 1;

    float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
    const float bx[4] = {float(sh.bbox.x1), float(sh.bbox.x2), float(sh.bbox.x2),
                         float(sh.bbox.x1)};
    const float by[4] = {float(sh.bbox.y1), float(sh.bbox.y1), float(sh.bbox.y2),
                         float(sh.bbox.y2)};

    for (int s = 0; s < d.count; ++s) {
      // Samples straddle the frame time: offset runs over [-shutter/2,
      // +shutter/2], taken at sample CENTRES so the average is unbiased. Taking
      // them from 0 forward instead would drag every piece half a shutter late.
      float offset = 0.0f;
      if (d.count > 1) {
        const float u = (float(s) + 0.5f) / float(d.count);  // (0,1)
        offset = (u - 0.5f) * shutter;
      }

      // Perspective is evaluated per shutter sample, so a piece rushing at the
      // camera grows through the smear rather than being smeared at one size.
      const float zAt = p.z + p.vz * offset;
      const float persp = PerspectiveScale(zAt, depth, fs);

      Mat3 fwd = ShardTransform(sh.cx, sh.cy, p.x + p.vx * offset, p.y + p.vy * offset,
                                p.angle + p.w * offset, p.scale * persp);

      // The card flip. Foreshortening a flat piece across an in-plane axis is a
      // squash in a rotated frame, which is a full 2x2 -- not something the
      // uniform scale above can express, hence the separate sandwich.
      //
      // Clamped away from zero: exactly edge-on the transform is singular, and
      // Invert would hand back identity, flashing the piece at full size for one
      // frame in the middle of its flip.
      if (p.tumble != 0.0f) {
        const float squash = fabsf(cosf(p.tumble));
        const Mat3 pivot = Mat3::Translate(sh.cx, sh.cy);
        const Mat3 unpivot = Mat3::Translate(-sh.cx, -sh.cy);
        fwd = fwd * pivot * Mat3::Rotate(p.tumbleAxis) *
              Mat3::Scale(1.0f, squash < 0.02f ? 0.02f : squash) *
              Mat3::Rotate(-p.tumbleAxis) * unpivot;
      }

      d.inv[s] = Invert(fwd);

      // Destination bounds must cover every sample, or the last one gets clipped
      // and the blur ends in a hard edge.
      for (int c = 0; c < 4; ++c) {
        float tx, ty;
        fwd.Apply(bx[c], by[c], tx, ty);
        minX = std::min(minX, tx);
        minY = std::min(minY, ty);
        maxX = std::max(maxX, tx);
        maxY = std::max(maxY, ty);
      }
    }

    // Grown by one pixel because the bilinear tap of an edge pixel reaches half
    // a pixel past it, and by another because the stencil is tested at pixel
    // centres.
    d.x1 = std::max(0, int(std::floor(minX)) - 2);
    d.y1 = std::max(0, int(std::floor(minY)) - 2);
    d.x2 = std::min(geom.width, int(std::ceil(maxX)) + 2);
    d.y2 = std::min(geom.height, int(std::ceil(maxY)) + 2);

    if (d.x2 <= d.x1 || d.y2 <= d.y1) continue;  // entirely off frame
    out.push_back(d);
  }

  // Painter's algorithm: furthest first, so nearer pieces cover them. Occlusion
  // is the strongest depth cue there is -- without it, perspective scaling alone
  // reads as pieces changing size rather than as pieces moving in depth.
  //
  // The tie-break on label keeps the order total. Without it, the thousands of
  // pieces sitting at z = 0 before the shatter starts would be ordered by
  // whatever the sort happened to do, and could reorder between frames -- which
  // shows up as flickering wherever two pieces overlap by a pixel.
  std::stable_sort(out.begin(), out.end(), [](const ShardDraw& a, const ShardDraw& b) {
    if (a.z != b.z) return a.z > b.z;
    return a.label < b.label;
  });
}

void BuildTileGrid(const std::vector<ShardDraw>& draws, int width, int height, TileGrid& out) {
  out.tilesX = (width + kTileSize - 1) / kTileSize;
  out.tilesY = (height + kTileSize - 1) / kTileSize;
  const int nt = out.tileCount();

  out.start.assign(std::size_t(nt) + 1, 0);
  out.shardIdx.clear();
  if (nt == 0 || draws.empty()) return;

  // Count, prefix sum, fill: the standard two-pass CSR build. Filling in draw
  // order gives each tile its shards in ascending index, which is what makes the
  // composite order deterministic and identical on the GPU.
  auto tileRange = [&](const ShardDraw& d, int& tx1, int& ty1, int& tx2, int& ty2) {
    tx1 = d.x1 / kTileSize;
    ty1 = d.y1 / kTileSize;
    tx2 = (d.x2 - 1) / kTileSize;
    ty2 = (d.y2 - 1) / kTileSize;
    tx1 = std::max(0, tx1);
    ty1 = std::max(0, ty1);
    tx2 = std::min(out.tilesX - 1, tx2);
    ty2 = std::min(out.tilesY - 1, ty2);
  };

  for (const ShardDraw& d : draws) {
    int tx1, ty1, tx2, ty2;
    tileRange(d, tx1, ty1, tx2, ty2);
    for (int ty = ty1; ty <= ty2; ++ty)
      for (int tx = tx1; tx <= tx2; ++tx) ++out.start[std::size_t(ty * out.tilesX + tx) + 1];
  }

  for (int i = 0; i < nt; ++i) out.start[std::size_t(i) + 1] += out.start[std::size_t(i)];
  out.shardIdx.resize(std::size_t(out.start[std::size_t(nt)]));

  std::vector<int32_t> cursor(out.start.begin(), out.start.end() - 1);
  for (std::size_t i = 0; i < draws.size(); ++i) {
    int tx1, ty1, tx2, ty2;
    tileRange(draws[i], tx1, ty1, tx2, ty2);
    for (int ty = ty1; ty <= ty2; ++ty)
      for (int tx = tx1; tx <= tx2; ++tx)
        out.shardIdx[std::size_t(cursor[std::size_t(ty * out.tilesX + tx)]++)] = int32_t(i);
  }
}

void Composite(const ImageView& dst, const ConstImageView& src, const ShatterGeometry& geom,
               const std::vector<ShardDraw>& draws, const TileGrid& tiles,
               const CompositeParams& params) {
  if (!dst.valid()) return;

  const SampleView sv = MakeSampleView(src);
  const int32_t* labels = geom.labelImage.data();

  // Null when the bevel is off, which is also how the shading is skipped.
  const bool haveEdges = params.bevel > 0.0f && geom.edges.valid() &&
                         geom.edges.width == geom.width && geom.edges.height == geom.height;
  const uint8_t* eDist = haveEdges ? geom.edges.dist.data() : nullptr;
  const uint8_t* eAngle = haveEdges ? geom.edges.angle.data() : nullptr;

  // Rows are independent because this is a gather: each destination pixel reads
  // and writes only itself, so the banding is invisible in the output.
  ParallelFor(dst.height, [&](int yBegin, int yEnd) {
    for (int y = yBegin; y < yEnd; ++y) {
      float* drow = dst.row(y);
      std::memset(drow, 0, std::size_t(dst.width) * 4 * sizeof(float));

      const int ty = y / kTileSize;
      if (ty >= tiles.tilesY) continue;

      for (int x = 0; x < dst.width; ++x) {
        const int tx = x / kTileSize;
        if (tx >= tiles.tilesX) continue;

        const int tile = ty * tiles.tilesX + tx;
        const int32_t lo = tiles.start[std::size_t(tile)];
        const int32_t hi = tiles.start[std::size_t(tile) + 1];
        if (lo == hi) continue;

        // The identical function the CUDA kernel calls.
        ShadePixel(labels, eDist, eAngle, geom.width, geom.height, sv, draws.data(),
                   tiles.shardIdx.data(), lo, hi, x, y, params.edgeAntialias,
                   params.showPieces, haveEdges ? params.bevel : 0.0f, params.bevelWidth,
                   params.lightX, params.lightY, drow + std::size_t(x) * 4);
      }
    }
  });
}

}  // namespace shp
