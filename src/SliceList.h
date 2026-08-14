// Reading and writing the hand-drawn cuts as a string parameter.
//
// A string rather than a pile of numbered Double2D parameters, for the reason
// AutoMask stores its selection anchors the same way: the count is not known in
// advance. Twenty pre-declared slice parameters would be twenty rows of clutter
// in the Inspector on the nineteen occasions nobody uses them, and a hard
// ceiling on the twentieth.
//
// The format is deliberately plain and lenient -- "x1,y1,x2,y2;x1,y1,x2,y2" in
// normalized frame coordinates -- so it survives being read by a human, edited
// by hand, or half-typed into the field without throwing mid-render.
#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "core/Cells.h"

namespace shp {

/// Beyond this the Inspector field becomes unreadable and the cost of rasterising
/// every cut starts to show. Nothing about the algorithm breaks above it; it is
/// a limit on the interface, not on the maths.
constexpr int kMaxSlices = 64;

/// Obstacles are tested against every moving piece every substep, so the ceiling
/// is lower than for cuts -- which are rasterised once and then forgotten.
constexpr int kMaxColliders = 32;

/** @brief Parse a list of segments from text.
 *
 * Templated because two different things are stored this way -- the cuts that
 * decide how the image breaks, and the colliders the simulation bounces off.
 * They are separate concepts with separate parameters, but the text format is
 * the same four numbers, and one parser that both use cannot drift.
 */
template <class T>
inline std::vector<T> ParseSegments(const std::string& in, int limit) {
  std::vector<T> out;

  // Any run of characters that is not part of a number separates, so commas,
  // semicolons, spaces and newlines all behave the same and a trailing
  // separator is harmless.
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int have = 0;
  std::size_t i = 0;

  while (i <= in.size() && int(out.size()) < limit) {
    // Skip anything that cannot begin a number.
    while (i < in.size() && in[i] != '-' && in[i] != '+' && in[i] != '.' &&
           !(in[i] >= '0' && in[i] <= '9'))
      ++i;
    if (i >= in.size()) break;

    char* end = nullptr;
    const float f = std::strtof(in.c_str() + i, &end);
    if (end == in.c_str() + i) {
      ++i;  // not actually a number after all
      continue;
    }
    i = std::size_t(end - in.c_str());

    v[have++] = f;
    if (have == 4) {
      T s;
      s.x1 = v[0];
      s.y1 = v[1];
      s.x2 = v[2];
      s.y2 = v[3];
      out.push_back(s);
      have = 0;
    }
  }
  return out;  // a trailing partial group is dropped, which is what half-typed means
}

template <class T>
inline std::string FormatSegments(const std::vector<T>& segs) {
  std::string out;
  char buf[96];
  for (std::size_t i = 0; i < segs.size(); ++i) {
    const T& s = segs[i];
    // Four decimals is about a fifth of a pixel at UHD -- finer than anyone can
    // drag, and short enough that a dozen entries still fit in a readable field.
    std::snprintf(buf, sizeof(buf), "%s%.4f,%.4f,%.4f,%.4f", i ? "; " : "", s.x1, s.y1, s.x2,
                  s.y2);
    out += buf;
  }
  return out;
}

inline std::vector<Slice> ParseSlices(const std::string& in) {
  return ParseSegments<Slice>(in, kMaxSlices);
}

inline std::string FormatSlices(const std::vector<Slice>& slices) {
  return FormatSegments(slices);
}

}  // namespace shp
