// How far every pixel is from the edge of its own piece, and which way that
// edge faces.
//
// This is what makes a flat cutout read as something with thickness. Shading a
// band along each piece's cut edges -- lit on the side facing the light, dark on
// the side away from it -- is the single strongest depth cue available without
// actually rendering geometry, because it is the cue a real bevelled edge gives.
//
// Two bytes per pixel, both quantised, because this sits alongside a label image
// that is already four:
//
//   dist   pixels to the nearest edge of this piece, clamped at 255
//   angle  direction of the INWARD normal, 0..255 mapping to 0..2pi
//
// The angle is stored rather than derived at shading time so the inner loop does
// no neighbourhood sampling: it reads two bytes, adds the shard's rotation, and
// takes one cosine.
//
// Distances are measured with a vector transform (Danielsson) rather than a
// chamfer. A chamfer's distances are close enough, but its GRADIENTS are blocky
// along diagonals, and the gradient is exactly what is being stored -- the
// artefacts would show up as facets running across every bevel.
#pragma once

#include "ImageView.h"

#include <cstdint>
#include <vector>

namespace shp {

struct EdgeField {
  int width = 0, height = 0;
  std::vector<uint8_t> dist;   ///< distance to this piece's edge, in pixels
  std::vector<uint8_t> angle;  ///< inward normal direction, 0..255 == 0..2pi

  bool valid() const { return !dist.empty() && dist.size() == angle.size(); }
};

/** @brief Build the field for every piece in @p labels at once.
 *
 * Propagation never crosses a label boundary, so each piece gets its own
 * interior distance even though the pieces tile the frame with no gaps between
 * them. That is the whole reason this cannot be a single global distance
 * transform of the alpha: the seam where two pieces meet has to read as an edge
 * for both of them, and a transform over the union would see no edge there at
 * all.
 */
void BuildEdgeField(const std::vector<int32_t>& labels, int width, int height, EdgeField& out);

}  // namespace shp
