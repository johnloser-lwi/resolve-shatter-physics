#include "ShardGeom.h"

#include <algorithm>

namespace shp {

namespace {

// Positive when the turn o->a->b is counter-clockwise.
float CrossOf(const Vec2& o, const Vec2& a, const Vec2& b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

}  // namespace

std::vector<Vec2> ConvexHull(std::vector<Vec2> pts) {
  const std::size_t n = pts.size();
  if (n < 3) return {};

  std::sort(pts.begin(), pts.end(), [](const Vec2& a, const Vec2& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  pts.erase(std::unique(pts.begin(), pts.end(),
                        [](const Vec2& a, const Vec2& b) { return a.x == b.x && a.y == b.y; }),
            pts.end());
  if (pts.size() < 3) return {};

  std::vector<Vec2> hull(pts.size() * 2);
  std::size_t k = 0;

  // Lower hull, then upper. The `>` in the pop test (rather than `>=`) keeps
  // collinear points on the hull; dropping them would be tidier but costs an
  // edge case where a shard that happens to be a perfect rectangle loses a
  // corner and the solver sees a triangle.
  for (std::size_t i = 0; i < pts.size(); ++i) {
    while (k >= 2 && CrossOf(hull[k - 2], hull[k - 1], pts[i]) <= 0.0f) --k;
    hull[k++] = pts[i];
  }
  const std::size_t lower = k + 1;
  for (std::size_t i = pts.size() - 1; i-- > 0;) {
    while (k >= lower && CrossOf(hull[k - 2], hull[k - 1], pts[i]) <= 0.0f) --k;
    hull[k++] = pts[i];
  }

  hull.resize(k > 0 ? k - 1 : 0);  // last point equals the first
  if (hull.size() < 3) return {};
  return hull;
}

float PolygonArea(const std::vector<Vec2>& poly) {
  const std::size_t n = poly.size();
  if (n < 3) return 0.0f;

  float acc = 0.0f;
  for (std::size_t i = 0, j = n - 1; i < n; j = i++) acc += Cross(poly[j], poly[i]);
  return acc * 0.5f;
}

Vec2 PolygonCentroid(const std::vector<Vec2>& poly) {
  const std::size_t n = poly.size();
  if (n == 0) return Vec2(0.0f, 0.0f);
  if (n < 3) {
    Vec2 acc(0.0f, 0.0f);
    for (const Vec2& p : poly) acc += p;
    return acc * (1.0f / float(n));
  }

  float a2 = 0.0f;
  Vec2 acc(0.0f, 0.0f);
  for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
    const float c = Cross(poly[j], poly[i]);
    a2 += c;
    acc += (poly[j] + poly[i]) * c;
  }

  // A zero-area polygon has no area centroid; fall back to the vertex mean
  // rather than dividing by zero and handing NaN to the solver.
  if (a2 > -1e-9f && a2 < 1e-9f) {
    Vec2 mean(0.0f, 0.0f);
    for (const Vec2& p : poly) mean += p;
    return mean * (1.0f / float(n));
  }
  return acc * (1.0f / (3.0f * a2));
}

float PolygonInertiaPerMass(const std::vector<Vec2>& poly, const Vec2& centroid) {
  const std::size_t n = poly.size();
  if (n < 3) return 0.0f;

  // Standard polygon second-moment sum, accumulated about the centroid so the
  // parallel-axis term is already gone.
  float num = 0.0f;
  float den = 0.0f;
  for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
    const Vec2 a = poly[j] - centroid;
    const Vec2 b = poly[i] - centroid;
    const float c = Cross(a, b);
    num += c * (Dot(a, a) + Dot(a, b) + Dot(b, b));
    den += c;
  }
  if (den > -1e-9f && den < 1e-9f) return 0.0f;

  const float v = num / (6.0f * den);
  return v > 0.0f ? v : 0.0f;
}

std::vector<Vec2> ClipHalfPlane(const std::vector<Vec2>& poly, const Vec2& n, float d) {
  const std::size_t cnt = poly.size();
  if (cnt < 3) return {};

  std::vector<Vec2> out;
  out.reserve(cnt + 4);

  for (std::size_t i = 0, j = cnt - 1; i < cnt; j = i++) {
    const Vec2& A = poly[j];
    const Vec2& B = poly[i];
    const float da = Dot(n, A) - d;
    const float db = Dot(n, B) - d;
    const bool inA = da <= 0.0f;
    const bool inB = db <= 0.0f;

    if (inA != inB) {
      // Crossing the plane. The denominator cannot vanish here: da and db have
      // opposite signs, so their difference is bounded away from zero.
      const float t = da / (da - db);
      out.push_back(Vec2(A.x + (B.x - A.x) * t, A.y + (B.y - A.y) * t));
    }
    if (inB) out.push_back(B);
  }

  if (out.size() < 3) return {};
  return out;
}

}  // namespace shp
