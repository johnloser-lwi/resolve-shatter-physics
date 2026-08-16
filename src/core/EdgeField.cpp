#include "EdgeField.h"

#include "TransformMath.h"

#include <algorithm>
#include <cmath>

namespace shp {
namespace {

// Offset from a pixel to the nearest edge, carried through the sweeps.
struct Offset {
  int16_t dx, dy;
};

constexpr int16_t kFar = 12000;  // squares to 1.44e8, comfortably inside int32

inline int dist2(const Offset& o) { return int(o.dx) * int(o.dx) + int(o.dy) * int(o.dy); }

}  // namespace

void BuildEdgeField(const std::vector<int32_t>& labels, int width, int height, EdgeField& out) {
  out.width = width;
  out.height = height;
  const std::size_t n = std::size_t(width) * std::size_t(height);
  out.dist.assign(n, 0);
  out.angle.assign(n, 0);
  if (n == 0 || labels.size() != n) return;

  std::vector<Offset> grid(n, Offset{kFar, kFar});

  // Seeds: any pixel of a piece that touches something which is not that piece.
  // Frame borders count, so a piece running off the edge is bevelled there too
  // rather than appearing to continue past the frame.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t p = std::size_t(y) * std::size_t(width) + std::size_t(x);
      const int32_t l = labels[p];
      if (l == 0) {
        grid[p] = Offset{0, 0};  // background is edge everywhere
        continue;
      }
      const bool edge = x == 0 || y == 0 || x == width - 1 || y == height - 1 ||
                        labels[p - 1] != l || labels[p + 1] != l ||
                        labels[p - std::size_t(width)] != l ||
                        labels[p + std::size_t(width)] != l;
      if (edge) grid[p] = Offset{0, 0};
    }
  }

  // Two sweeps, forward then backward, each relaxing against the neighbours it
  // has already visited. Propagation is confined to matching labels, which is
  // what keeps one piece's interior from leaking into the piece next to it.
  //
  // The step passed in is the vector FROM the neighbour TO this pixel, and its
  // sign matters as much as its size. Getting a sign wrong still yields the
  // right distance -- the length is unchanged -- but flips the stored direction,
  // so the bevel lights from the wrong side on whichever edges that neighbour
  // happened to win. Distances looking correct is no evidence the directions are.
  auto relax = [&](std::size_t p, std::size_t q, int sx, int sy, int32_t label) {
    if (labels[q] != label) return;
    int cx = int(grid[q].dx) + sx;
    int cy = int(grid[q].dy) + sy;
    if (cx > kFar) cx = kFar;
    if (cy > kFar) cy = kFar;
    if (cx < -kFar) cx = -kFar;
    if (cy < -kFar) cy = -kFar;
    const Offset cand{int16_t(cx), int16_t(cy)};
    if (dist2(cand) < dist2(grid[p])) grid[p] = cand;
  };

  const std::size_t W = std::size_t(width);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t p = std::size_t(y) * W + std::size_t(x);
      const int32_t l = labels[p];
      if (l == 0) continue;
      if (y > 0) relax(p, p - W, 0, 1, l);                            // from above
      if (x > 0) relax(p, p - 1, 1, 0, l);                            // from the left
      if (x > 0 && y > 0) relax(p, p - W - 1, 1, 1, l);               // up-left
      if (x < width - 1 && y > 0) relax(p, p - W + 1, -1, 1, l);      // up-right
    }
    for (int x = width - 1; x >= 0; --x) {
      const std::size_t p = std::size_t(y) * W + std::size_t(x);
      const int32_t l = labels[p];
      if (l == 0) continue;
      if (x < width - 1) relax(p, p + 1, -1, 0, l);                   // from the right
    }
  }

  for (int y = height - 1; y >= 0; --y) {
    for (int x = width - 1; x >= 0; --x) {
      const std::size_t p = std::size_t(y) * W + std::size_t(x);
      const int32_t l = labels[p];
      if (l == 0) continue;
      if (y < height - 1) relax(p, p + W, 0, -1, l);                  // from below
      if (x < width - 1) relax(p, p + 1, -1, 0, l);                   // from the right
      if (x < width - 1 && y < height - 1) relax(p, p + W + 1, -1, -1, l);  // down-right
      if (x > 0 && y < height - 1) relax(p, p + W - 1, 1, -1, l);     // down-left
    }
    for (int x = 0; x < width; ++x) {
      const std::size_t p = std::size_t(y) * W + std::size_t(x);
      const int32_t l = labels[p];
      if (l == 0) continue;
      if (x > 0) relax(p, p - 1, 1, 0, l);                            // from the left
    }
  }

  // Quantise.
  //
  // The offset accumulates as steps taken AWAY from a seed, so it is already the
  // vector from the nearest edge to the pixel -- which is the inward direction,
  // and needs no negating. Negating it here gives an outward normal, which lights
  // every bevel from precisely the wrong side: subtle enough to look like a
  // lighting preference rather than a bug.
  const float toByte = 255.0f / (2.0f * kPi);
  for (std::size_t p = 0; p < n; ++p) {
    if (labels[p] == 0) continue;
    const Offset& o = grid[p];
    if (o.dx >= kFar || o.dy >= kFar) continue;

    const float d = std::sqrt(float(dist2(o)));
    out.dist[p] = uint8_t(std::min(255.0f, d));

    // A seed pixel has no direction of its own; borrowing the sign of a zero
    // vector would give atan2(0,0) and a hard seam right along every edge.
    // Its neighbours resolve it, and at distance 0 the shading is at full
    // strength anyway, so the angle barely matters there.
    float a = (o.dx == 0 && o.dy == 0) ? 0.0f : atan2f(float(o.dy), float(o.dx));
    if (a < 0.0f) a += 2.0f * kPi;
    out.angle[p] = uint8_t(std::min(255.0f, a * toByte));
  }
}

}  // namespace shp
