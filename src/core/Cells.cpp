#include "Cells.h"

#include "Hash.h"
#include "Segmentation.h"

#include <algorithm>
#include <cmath>

namespace shp {
namespace {

// How far out to look for a nearer seed, and for a bisector to clip against.
//
// A jittered seed can leave its own grid cell, so the nearest seed to a pixel is
// not always in the 3x3 block around it. Two rings covers the worst case at full
// jitter, where a seed can stray half a cell in each axis.
int searchRadiusFor(float jitter) { return jitter > 0.5f ? 2 : 1; }

struct VoronoiGrid {
  int gx = 1, gy = 1;
  float jitter = 0.0f;
  uint32_t seed = 1;
  int w = 1, h = 1;

  int count() const { return gx * gy; }

  // Seed position in PIXELS. Distances are measured in pixel space rather than
  // in normalized space so that cells come out round on a 16:9 frame instead of
  // stretched into lozenges.
  Vec2 seedAt(int i, int j) const {
    const uint32_t idx = uint32_t(j * gx + i);
    const float jx = (Hash01(idx, seed, kChanSeedX) - 0.5f) * jitter;
    const float jy = (Hash01(idx, seed, kChanSeedY) - 0.5f) * jitter;
    const float nx = (float(i) + 0.5f + jx) / float(gx);
    const float ny = (float(j) + 0.5f + jy) / float(gy);
    return Vec2(nx * float(w), ny * float(h));
  }
};

VoronoiGrid makeVoronoiGrid(int w, int h, const ShapeParams& p) {
  VoronoiGrid g;
  g.w = w;
  g.h = h;
  g.jitter = Clamp01(p.cellJitter);
  g.seed = uint32_t(p.shapeSeed);

  // Aspect-aware split of the requested cell count, so cells stay roughly square.
  const float target = float(std::max(1, p.cellCount));
  const float aspect = float(w) / float(std::max(1, h));
  g.gx = std::max(1, int(std::lround(std::sqrt(target * aspect))));
  g.gy = std::max(1, int(std::lround(target / float(g.gx))));
  return g;
}

struct BrickGrid {
  int rows = 1;
  int cols = 1;
  float offset = 0.0f;
  int w = 1, h = 1;

  int count() const { return rows * cols + rows; }  // +rows for the wrapped stub per row
};

BrickGrid makeBrickGrid(int w, int h, const ShapeParams& p, bool stagger) {
  BrickGrid g;
  g.w = w;
  g.h = h;
  g.rows = std::max(1, p.brickRows);
  g.offset = stagger ? p.brickOffset : 0.0f;

  // Bricks default to roughly twice as wide as they are tall, which is what
  // makes them read as masonry rather than as a grid.
  const float rowH = float(h) / float(g.rows);
  const float brickW = stagger ? rowH * 2.0f : rowH;
  g.cols = std::max(1, int(std::lround(float(w) / std::max(1.0f, brickW))));
  return g;
}

}  // namespace

// ---------------------------------------------------------------------------

std::vector<Vec2> voronoiCellPolygon(int cellIndex, int w, int h, const ShapeParams& params) {
  const VoronoiGrid g = makeVoronoiGrid(w, h, params);
  if (cellIndex < 0 || cellIndex >= g.count()) return {};

  const int ci = cellIndex % g.gx;
  const int cj = cellIndex / g.gx;
  const Vec2 S = g.seedAt(ci, cj);

  // Start from the whole frame and cut once per nearby seed. The result is the
  // true Voronoi cell clipped to the frame: exact, convex by construction, and
  // free of the staircase a raster boundary trace would produce.
  std::vector<Vec2> poly = {Vec2(0.0f, 0.0f), Vec2(float(w), 0.0f), Vec2(float(w), float(h)),
                            Vec2(0.0f, float(h))};

  const int R = searchRadiusFor(g.jitter) + 1;
  for (int dj = -R; dj <= R; ++dj) {
    for (int di = -R; di <= R; ++di) {
      if (di == 0 && dj == 0) continue;
      const int ni = ci + di, nj = cj + dj;
      if (ni < 0 || nj < 0 || ni >= g.gx || nj >= g.gy) continue;

      const Vec2 T = g.seedAt(ni, nj);
      // Points closer to S than to T satisfy dot(p, T-S) <= (|T|^2 - |S|^2)/2.
      const Vec2 n = T - S;
      const float d = (Dot(T, T) - Dot(S, S)) * 0.5f;
      poly = ClipHalfPlane(poly, n, d);
      if (poly.size() < 3) return {};
    }
  }
  return poly;
}

std::vector<int32_t> buildCellMap(int w, int h, const ShapeParams& params, int& cellCountOut) {
  cellCountOut = 0;
  const std::size_t n = std::size_t(w) * std::size_t(h);
  std::vector<int32_t> cells(n, 1);

  switch (params.pattern) {
    case Pattern::Voronoi: {
      const VoronoiGrid g = makeVoronoiGrid(w, h, params);
      const int R = searchRadiusFor(g.jitter);
      cellCountOut = g.count();

      // Seeds are cheap to recompute but not free, and every pixel looks at up
      // to 25 of them; caching the whole grid once turns the inner loop into
      // pure arithmetic.
      std::vector<Vec2> seeds(std::size_t(g.count()));
      for (int j = 0; j < g.gy; ++j)
        for (int i = 0; i < g.gx; ++i) seeds[std::size_t(j * g.gx + i)] = g.seedAt(i, j);

      for (int y = 0; y < h; ++y) {
        const float py = float(y) + 0.5f;
        const int j0 = std::min(g.gy - 1, std::max(0, int(float(y) * float(g.gy) / float(h))));
        for (int x = 0; x < w; ++x) {
          const float px = float(x) + 0.5f;
          const int i0 = std::min(g.gx - 1, std::max(0, int(float(x) * float(g.gx) / float(w))));

          int best = -1;
          float bestD = 0.0f;
          for (int dj = -R; dj <= R; ++dj) {
            const int nj = j0 + dj;
            if (nj < 0 || nj >= g.gy) continue;
            for (int di = -R; di <= R; ++di) {
              const int ni = i0 + di;
              if (ni < 0 || ni >= g.gx) continue;
              const int idx = nj * g.gx + ni;
              const Vec2& s = seeds[std::size_t(idx)];
              const float dx = s.x - px, dy = s.y - py;
              const float d = dx * dx + dy * dy;
              if (best < 0 || d < bestD) {
                best = idx;
                bestD = d;
              }
            }
          }
          cells[std::size_t(y) * std::size_t(w) + std::size_t(x)] = best + 1;
        }
      }
      break;
    }

    case Pattern::Bricks:
    case Pattern::Grid: {
      const bool stagger = params.pattern == Pattern::Bricks;
      const BrickGrid g = makeBrickGrid(w, h, params, stagger);
      cellCountOut = g.count();

      const float rowH = float(h) / float(g.rows);
      const float colW = float(w) / float(g.cols);

      for (int y = 0; y < h; ++y) {
        const int row = std::min(g.rows - 1, int(float(y) / rowH));
        // Alternate rows slide sideways. Shifting by a fraction of a column and
        // then flooring means the row's first brick is a partial one, which is
        // exactly what a real running bond looks like at the wall's edge.
        const float shift = (row % 2) ? g.offset * colW : 0.0f;
        for (int x = 0; x < w; ++x) {
          float fx = (float(x) + 0.5f - shift) / colW;
          if (fx < 0.0f) fx = -1.0f;  // the stub left of the first full brick
          const int col = std::min(g.cols, int(std::floor(fx)) + 1);
          cells[std::size_t(y) * std::size_t(w) + std::size_t(x)] =
              row * (g.cols + 1) + std::max(0, col) + 1;
        }
      }
      break;
    }

    case Pattern::Alpha:
    case Pattern::Map:
      // No generated pattern: connectivity comes from the alpha itself, or from
      // the map image, which the caller supplies as keys.
      cellCountOut = 1;
      break;
  }

  return cells;
}

// ---------------------------------------------------------------------------

namespace {

// Quantised RGB of the map input, packed into one int. Regions of equal value
// become pieces.
//
// Quantising all three channels rather than luma alone is deliberate: two
// regions painted different hues at the same brightness are obviously different
// pieces to whoever drew the map, and a luma-only key would silently weld them
// together.
std::vector<int32_t> buildMapKeys(const ConstImageView& map, int w, int h, int levels) {
  std::vector<int32_t> keys(std::size_t(w) * std::size_t(h), 0);
  const float scale = float(std::max(2, levels) - 1);

  for (int y = 0; y < h; ++y) {
    // The map may be a different size from the source; sample it proportionally
    // rather than refusing to run.
    const int my = map.height == h ? y : std::min(map.height - 1, y * map.height / std::max(1, h));
    const float* row = map.row(my);
    for (int x = 0; x < w; ++x) {
      const int mx = map.width == w ? x : std::min(map.width - 1, x * map.width / std::max(1, w));
      const float* px = row + std::size_t(mx) * 4;
      const int r = int(Clamp01(px[0]) * scale + 0.5f);
      const int g = int(Clamp01(px[1]) * scale + 0.5f);
      const int b = int(Clamp01(px[2]) * scale + 0.5f);
      keys[std::size_t(y) * std::size_t(w) + std::size_t(x)] = (r << 16) | (g << 8) | b;
    }
  }
  return keys;
}

// Boundary pixels of each shard, which is all the convex hull needs.
//
// Hulling every pixel of a large shard would be quadratic in the worst case and
// wasteful in every case: an interior pixel can never be a hull vertex. A pixel
// is on the boundary when any 4-neighbour carries a different label, which is
// one pass over the image and yields O(perimeter) points.
void collectBoundaries(const std::vector<int32_t>& labels, int w, int h, int shardCount,
                       std::vector<std::vector<Vec2>>& out) {
  out.assign(std::size_t(shardCount), {});

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t p = std::size_t(y) * std::size_t(w) + std::size_t(x);
      const int32_t l = labels[p];
      if (l == 0) continue;

      // The frame-edge tests come first and short-circuit, which is what keeps
      // the four neighbour reads below in bounds.
      const bool edge =
          x == 0 || y == 0 || x == w - 1 || y == h - 1 ||
          labels[p - 1] != l || labels[p + 1] != l ||
          labels[p - std::size_t(w)] != l || labels[p + std::size_t(w)] != l;
      if (!edge) continue;

      out[std::size_t(l - 1)].push_back(Vec2(float(x) + 0.5f, float(y) + 0.5f));
    }
  }
}

// Cap the number of points fed to the hull. Monotone chain is O(n log n) and a
// UHD shard can have tens of thousands of boundary pixels; taking every k-th
// changes the hull by at most a pixel or two, which no collision is sensitive to.
constexpr std::size_t kMaxHullInput = 512;

std::vector<Vec2> subsampled(const std::vector<Vec2>& pts) {
  if (pts.size() <= kMaxHullInput) return pts;
  const std::size_t stride = (pts.size() + kMaxHullInput - 1) / kMaxHullInput;
  std::vector<Vec2> out;
  out.reserve(pts.size() / stride + 1);
  for (std::size_t i = 0; i < pts.size(); i += stride) out.push_back(pts[i]);
  return out;
}

}  // namespace

std::shared_ptr<const ShatterGeometry> buildGeometry(const ConstImageView& src,
                                                     const ConstImageView* map,
                                                     const ShapeParams& params) {
  auto geom = std::make_shared<ShatterGeometry>();
  if (!src.valid()) return geom;

  const int w = src.width, h = src.height;
  geom->width = w;
  geom->height = h;

  // 1. Which pixels are part of the image at all.
  bool any = false;
  std::vector<uint8_t> mask = buildAlphaMask(src, params.alphaThreshold, any);
  if (!any) return geom;

  // 2. Optional bridging, so two pieces joined by a hairline stay one piece.
  //    Radius is a fraction of frame height for the resolution-independence
  //    reason at the top of this file.
  const int bridgePx = int(std::lround(double(params.bridgeRadius) * double(h)));
  std::vector<uint8_t> dilated;
  if (bridgePx > 0) dilated = dilateMask(mask, w, h, bridgePx);
  const std::vector<uint8_t>& connectMask = bridgePx > 0 ? dilated : mask;

  // 3. The pattern, as a per-pixel key. Two touching pixels only end up in the
  //    same piece when their keys agree, so this is where a generated pattern
  //    and the alpha stencil get intersected -- in one union-find pass rather
  //    than as a separate clipping step.
  std::vector<int32_t> keys;
  const std::vector<int32_t>* keyPtr = nullptr;

  ShapeParams effective = params;
  if (effective.pattern == Pattern::Map && (!map || !map->valid())) {
    // A missing map should look like a default, not like a broken effect.
    effective.pattern = Pattern::Voronoi;
  }

  if (effective.pattern == Pattern::Map) {
    keys = buildMapKeys(*map, w, h, effective.mapLevels);
    keyPtr = &keys;
  } else if (effective.pattern != Pattern::Alpha) {
    int cellCount = 0;
    keys = buildCellMap(w, h, effective, cellCount);
    keyPtr = &keys;
  }

  // 4. Label. Connectivity from the (possibly dilated) mask, ownership from the
  //    true one, so bounding boxes stay exact.
  LabelResult lr =
      labelComponents(connectMask, keyPtr, bridgePx > 0 ? &mask : nullptr, w, h);
  if (lr.count == 0) return geom;

  // 5. Despeckle. Relative to frame area, so the same setting means the same
  //    thing at proxy and at delivery.
  const int minArea = std::max(
      1, int(std::lround(double(params.minPieceArea) * double(w) * double(h))));
  despeckle(lr, minArea);
  if (lr.count == 0) return geom;

  // 6. Shards.
  geom->labelImage = std::move(lr.labels);
  geom->shards.resize(std::size_t(lr.count));
  for (int l = 1; l <= lr.count; ++l) {
    const ComponentStat& s = lr.stats[std::size_t(l)];
    Shard& sh = geom->shards[std::size_t(l - 1)];
    sh.id = l;
    sh.cx = s.centroidX();
    sh.cy = s.centroidY();
    sh.area = float(s.area);
    sh.bbox = s.bbox;
    // A usable fallback even without a polygon: half the bbox diagonal always
    // encloses the shard, so the broadphase stays correct in ballistic mode
    // where no hull is built.
    const float hw = 0.5f * float(s.bbox.width());
    const float hh = 0.5f * float(s.bbox.height());
    sh.radius = std::sqrt(hw * hw + hh * hh);
  }

  // 6b. Frame-level averages, so the render path never has to loop for them.
  {
    double area = 0.0, radius = 0.0;
    for (const Shard& s : geom->shards) {
      area += double(s.area);
      radius += double(s.radius);
    }
    const double n = double(geom->shards.size());
    geom->meanArea = n > 0.0 ? float(area / n) : 1.0f;
    geom->meanRadius = n > 0.0 ? float(radius / n) : 1.0f;
  }

  // 7. Collision proxies, only when someone is going to collide them.
  if (params.buildPolygons) {
    // Reads geom->labelImage rather than lr.labels: the labels were moved into
    // the geometry above, so lr.labels is an empty vector by this point.
    std::vector<std::vector<Vec2>> boundaries;
    collectBoundaries(geom->labelImage, w, h, lr.count, boundaries);

    for (int l = 1; l <= lr.count; ++l) {
      Shard& sh = geom->shards[std::size_t(l - 1)];
      sh.poly = ConvexHull(subsampled(boundaries[std::size_t(l - 1)]));
      if (sh.poly.size() < 3) {
        // Degenerate: a one-pixel-wide sliver has no usable hull. It keeps its
        // bbox radius and simply never generates contacts.
        sh.poly.clear();
        continue;
      }

      // The hull's own centroid, not the pixel centroid: the solver applies
      // impulses about the point the polygon actually turns around, and using
      // the pixel centroid of a shape with holes would put that point outside
      // the collision proxy.
      const Vec2 c = PolygonCentroid(sh.poly);
      sh.inertiaPerMass = PolygonInertiaPerMass(sh.poly, c);

      float rmax = 0.0f;
      for (const Vec2& v : sh.poly) {
        const float d = LengthSq(Vec2(v.x - sh.cx, v.y - sh.cy));
        if (d > rmax) rmax = d;
      }
      sh.radius = std::sqrt(rmax);
    }
  }

  return geom;
}

}  // namespace shp
