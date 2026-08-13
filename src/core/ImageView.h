// Host-independent view over a premultiplied float RGBA buffer.
//
// Everything under src/core works in terms of this, which is what lets the same
// cell-generation, simulation and compositing code run inside Resolve and inside
// tools/shattertest. src/core must never include an OFX header -- that rule is
// what keeps the offline harness linking the identical code Resolve runs, so
// tuning a shatter costs a second instead of a Resolve relaunch.
#pragma once

#include <cstddef>
#include <cstdint>

namespace shp {

struct RectI {
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // [x1,x2) [y1,y2)

  int width() const { return x2 - x1; }
  int height() const { return y2 - y1; }
  bool empty() const { return x2 <= x1 || y2 <= y1; }

  void grow(int px) {
    x1 -= px;
    y1 -= px;
    x2 += px;
    y2 += px;
  }

  void unionWith(const RectI& o) {
    if (o.empty()) return;
    if (empty()) {
      *this = o;
      return;
    }
    if (o.x1 < x1) x1 = o.x1;
    if (o.y1 < y1) y1 = o.y1;
    if (o.x2 > x2) x2 = o.x2;
    if (o.y2 > y2) y2 = o.y2;
  }

  void includePoint(int x, int y) {
    if (empty()) {
      x1 = x;
      y1 = y;
      x2 = x + 1;
      y2 = y + 1;
      return;
    }
    if (x < x1) x1 = x;
    if (y < y1) y1 = y;
    if (x >= x2) x2 = x + 1;
    if (y >= y2) y2 = y + 1;
  }
};

// A non-owning view. `data` points at the pixel whose coordinate is (0,0);
// rowStride is in floats, not bytes, and may be negative for bottom-up images.
struct ImageView {
  float* data = nullptr;
  int width = 0;
  int height = 0;
  std::ptrdiff_t rowStride = 0;  // in floats

  float* row(int y) const { return data + rowStride * y; }
  float* at(int x, int y) const { return row(y) + std::ptrdiff_t(x) * 4; }
  bool valid() const { return data != nullptr && width > 0 && height > 0; }
};

// Read-only twin. Kept separate rather than templating ImageView on constness:
// the sampling code in Sampler.h is device-safe and shared with CUDA, where a
// template over const would need explicit instantiation for no benefit.
struct ConstImageView {
  const float* data = nullptr;
  int width = 0;
  int height = 0;
  std::ptrdiff_t rowStride = 0;  // in floats

  ConstImageView() = default;
  ConstImageView(const ImageView& v)
      : data(v.data), width(v.width), height(v.height), rowStride(v.rowStride) {}

  const float* row(int y) const { return data + rowStride * y; }
  const float* at(int x, int y) const { return row(y) + std::ptrdiff_t(x) * 4; }
  bool valid() const { return data != nullptr && width > 0 && height > 0; }
};

}  // namespace shp
