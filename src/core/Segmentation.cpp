#include "Segmentation.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace shp {
namespace {

// ---------------------------------------------------------------- union-find

struct UnionFind {
  std::vector<int> parent;

  int add() {
    parent.push_back(int(parent.size()));
    return int(parent.size()) - 1;
  }

  int find(int a) {
    while (parent[a] != a) {
      parent[a] = parent[parent[a]];  // path halving
      a = parent[a];
    }
    return a;
  }

  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
  }
};

// ------------------------------------------------------------------ dilation

void dilateLine(const uint8_t* src, uint8_t* dst, int n, std::ptrdiff_t stride, int r,
                std::vector<int>& prev, std::vector<int>& next) {
  const int kFar = std::numeric_limits<int>::max() / 4;

  int last = -kFar;
  for (int i = 0; i < n; ++i) {
    if (src[i * stride]) last = i;
    prev[std::size_t(i)] = last;
  }
  int ahead = kFar;
  for (int i = n - 1; i >= 0; --i) {
    if (src[i * stride]) ahead = i;
    next[std::size_t(i)] = ahead;
  }
  for (int i = 0; i < n; ++i)
    dst[i * stride] = uint8_t((i - prev[std::size_t(i)] <= r) || (next[std::size_t(i)] - i <= r));
}

}  // namespace

// ---------------------------------------------------------------------------

std::vector<uint8_t> buildAlphaMask(const ConstImageView& src, float threshold, bool& anyOut) {
  anyOut = false;
  if (!src.valid()) return {};

  const int w = src.width, h = src.height;
  std::vector<uint8_t> mask(std::size_t(w) * std::size_t(h), 0);
  for (int y = 0; y < h; ++y) {
    const float* row = src.row(y);
    uint8_t* m = &mask[std::size_t(y) * std::size_t(w)];
    for (int x = 0; x < w; ++x) {
      if (row[x * 4 + 3] > threshold) {
        m[x] = 1;
        anyOut = true;
      }
    }
  }
  return mask;
}

std::vector<uint8_t> dilateMask(const std::vector<uint8_t>& src, int w, int h, int r) {
  if (r <= 0) return src;

  std::vector<uint8_t> tmp(std::size_t(w) * std::size_t(h), 0);
  std::vector<uint8_t> out(std::size_t(w) * std::size_t(h), 0);
  std::vector<int> prev(std::size_t(std::max(w, h))), next(std::size_t(std::max(w, h)));

  for (int y = 0; y < h; ++y)
    dilateLine(&src[std::size_t(y) * std::size_t(w)], &tmp[std::size_t(y) * std::size_t(w)], w, 1,
               r, prev, next);
  for (int x = 0; x < w; ++x)
    dilateLine(&tmp[std::size_t(x)], &out[std::size_t(x)], h, w, r, prev, next);
  return out;
}

LabelResult labelComponents(const std::vector<uint8_t>& mask,
                            const std::vector<int32_t>* keys,
                            const std::vector<uint8_t>* writeMask,
                            int w, int h) {
  LabelResult lr;
  lr.width = w;
  lr.height = h;
  if (w <= 0 || h <= 0) return lr;

  const std::size_t n = std::size_t(w) * std::size_t(h);

  // Provisional labels, resolved through the union-find at write-back time.
  std::vector<int32_t> prov(n, 0);
  UnionFind uf;
  uf.add();  // label 0 == background

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t p = std::size_t(y) * std::size_t(w) + std::size_t(x);
      if (!mask[p]) continue;

      const int32_t myKey = keys ? (*keys)[p] : 0;
      int best = 0;

      // Only the four already-visited neighbours need checking; the other four
      // will consider this pixel when their turn comes.
      auto consider = [&](int nx, int ny) {
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) return;
        const std::size_t q = std::size_t(ny) * std::size_t(w) + std::size_t(nx);
        const int32_t l = prov[q];
        if (l == 0) return;
        if (keys && (*keys)[q] != myKey) return;
        if (best == 0) {
          best = l;
        } else {
          uf.unite(best, l);
          best = std::min(uf.find(best), uf.find(l));
        }
      };
      consider(x - 1, y);
      consider(x - 1, y - 1);
      consider(x, y - 1);
      consider(x + 1, y - 1);

      prov[p] = best != 0 ? best : uf.add();
    }
  }

  // Compact roots to sequential labels.
  std::vector<int> remap(uf.parent.size(), 0);
  int labelCount = 0;
  for (std::size_t i = 1; i < uf.parent.size(); ++i)
    if (uf.find(int(i)) == int(i)) remap[i] = ++labelCount;
  if (labelCount == 0) return lr;

  lr.count = labelCount;
  lr.labels.assign(n, 0);
  lr.stats.assign(std::size_t(labelCount) + 1, ComponentStat{});

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t p = std::size_t(y) * std::size_t(w) + std::size_t(x);
      if (prov[p] == 0) continue;
      // Connectivity was decided on `mask`; ownership is decided here. With a
      // bridging dilation in play those are different images, and using the
      // dilated one for both would inflate every bounding box by the radius.
      if (writeMask && !(*writeMask)[p]) continue;

      const int lab = remap[uf.find(prov[p])];
      lr.labels[p] = lab;

      ComponentStat& c = lr.stats[std::size_t(lab)];
      if (c.area == 0) {
        c.bbox = RectI{x, y, x + 1, y + 1};
      } else {
        c.bbox.x1 = std::min(c.bbox.x1, x);
        c.bbox.y1 = std::min(c.bbox.y1, y);
        c.bbox.x2 = std::max(c.bbox.x2, x + 1);
        c.bbox.y2 = std::max(c.bbox.y2, y + 1);
      }
      // Pixel centres, so a one-pixel component's centroid lands in its middle
      // rather than on its top-left corner.
      c.sumX += double(x) + 0.5;
      c.sumY += double(y) + 0.5;
      ++c.area;
    }
  }

  return lr;
}

void despeckle(LabelResult& lr, int minArea) {
  if (lr.count == 0 || minArea <= 1) return;

  // Survivors get consecutive new numbers so that "label - 1" stays a valid
  // index into the shard vector.
  std::vector<int> remap(std::size_t(lr.count) + 1, 0);
  int kept = 0;
  for (int l = 1; l <= lr.count; ++l)
    if (lr.stats[std::size_t(l)].area >= minArea) remap[std::size_t(l)] = ++kept;

  if (kept == lr.count) return;  // nothing dropped, nothing to renumber

  for (std::size_t p = 0; p < lr.labels.size(); ++p) {
    const int32_t l = lr.labels[p];
    if (l != 0) lr.labels[p] = remap[std::size_t(l)];
  }

  std::vector<ComponentStat> compacted(std::size_t(kept) + 1, ComponentStat{});
  for (int l = 1; l <= lr.count; ++l) {
    const int nl = remap[std::size_t(l)];
    if (nl != 0) compacted[std::size_t(nl)] = lr.stats[std::size_t(l)];
  }
  lr.stats.swap(compacted);
  lr.count = kept;
}

uint64_t hashAlpha(const ConstImageView& src, float threshold) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a offset basis
  if (!src.valid()) return h;

  for (int y = 0; y < src.height; y += 8) {
    const float* row = src.row(y);
    for (int x = 0; x < src.width; x += 8) {
      const float a = row[x * 4 + 3];
      // Rounded, not truncated. Truncation makes bucket 15 reachable only by
      // exactly 1.0, so the most common value in an alpha channel sits alone on
      // a boundary and the tiniest dip below opaque -- a re-encode, a different
      // codec, a colour-managed round trip -- invalidates the cache on every
      // frame. Rounding puts 1.0 in the middle of its bucket like every other
      // value.
      const uint8_t q = uint8_t(std::min(15.0f, std::max(0.0f, a) * 15.0f + 0.5f));
      h = (h ^ uint64_t(q + (a > threshold ? 16 : 0))) * 1099511628211ull;
    }
  }
  h = (h ^ uint64_t(src.width)) * 1099511628211ull;
  h = (h ^ uint64_t(src.height)) * 1099511628211ull;
  return h;
}

uint64_t hashRGB(const ConstImageView& src) {
  uint64_t h = 1469598103934665603ull;
  if (!src.valid()) return h;

  for (int y = 0; y < src.height; y += 8) {
    const float* row = src.row(y);
    for (int x = 0; x < src.width; x += 8) {
      for (int c = 0; c < 3; ++c) {
        // Rounded for the same reason as the alpha hash above.
        const uint8_t q =
            uint8_t(std::min(15.0f, std::max(0.0f, row[x * 4 + c]) * 15.0f + 0.5f));
        h = (h ^ uint64_t(q)) * 1099511628211ull;
      }
    }
  }
  h = (h ^ uint64_t(src.width)) * 1099511628211ull;
  h = (h ^ uint64_t(src.height)) * 1099511628211ull;
  return h;
}

}  // namespace shp
