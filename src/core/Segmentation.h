// Turning a frame into regions: threshold, bridge, label, despeckle.
//
// Adapted from text-animator's src/segmentation.cpp, with every trace of
// typography removed -- no glyphs, no lines, no words, no italic deskew. What is
// left is the part that was always general: given a binary mask, find its
// connected regions and measure them.
//
// The one addition is the KEY image in labelComponents. Text Animator only ever
// needed "pixels touch, so they are one thing"; a shatter also needs "pixels
// touch AND belong to the same pattern cell, so they are one piece". Passing the
// cell id as a key expresses that with the same union-find rather than a second
// labeller that would drift out of step with this one.
#pragma once

#include "ImageView.h"

#include <cstdint>
#include <vector>

namespace shp {

struct ComponentStat {
  RectI bbox;            ///< tight pixel bounds
  double sumX = 0.0;     ///< running sums, so the centroid needs no second pass
  double sumY = 0.0;
  int area = 0;          ///< pixel count

  float centroidX() const { return area > 0 ? float(sumX / area) : 0.0f; }
  float centroidY() const { return area > 0 ? float(sumY / area) : 0.0f; }
};

struct LabelResult {
  int width = 0, height = 0;
  std::vector<int32_t> labels;   ///< 0 = background, otherwise 1..count
  std::vector<ComponentStat> stats;  ///< index 0 unused, so stats[label] reads directly
  int count = 0;
};

/** @brief Binary mask of pixels whose alpha exceeds @p threshold.
 *
 * @param anyOut set false when the frame is entirely transparent, which callers
 *               must treat as "nothing to shatter" rather than as one big piece. */
std::vector<uint8_t> buildAlphaMask(const ConstImageView& src, float threshold, bool& anyOut);

/** @brief Separable binary dilation. Cost does not depend on the radius.
 *
 * The obvious version scans the window at every pixel, so it slows down as the
 * radius grows -- felt exactly when dragging the Bridge slider, since every
 * change re-runs the whole analysis. Instead each line is swept twice, carrying
 * the position of the nearest set pixel behind and ahead; a pixel is set if
 * either is within r. Exact for a binary mask, and identical to the window scan
 * it replaces. */
std::vector<uint8_t> dilateMask(const std::vector<uint8_t>& src, int w, int h, int r);

/** @brief Connected components, 8-connectivity, two-pass with union-find.
 *
 * @param mask      nonzero means foreground
 * @param keys      optional, one int32 per pixel. When present, two neighbouring
 *                  foreground pixels join only if their keys are equal -- which
 *                  is how a pattern cell split into two alpha islands becomes two
 *                  separate pieces instead of one piece with a gap.
 * @param writeMask optional, one uint8 per pixel. When present, labels are
 *                  written back only where this is nonzero, while connectivity
 *                  is still decided by @p mask. That is how bridging works: the
 *                  dilated mask decides what joins, the true mask decides which
 *                  pixels the piece actually owns, so bounding boxes stay exact.
 */
LabelResult labelComponents(const std::vector<uint8_t>& mask,
                            const std::vector<int32_t>* keys,
                            const std::vector<uint8_t>* writeMask,
                            int w, int h);

/** @brief Drop components smaller than @p minArea and renumber the survivors.
 *
 * Renumbering matters: the compositor indexes shards by label - 1, so a gap in
 * the numbering would either skip a shard or read past the end of the vector. */
void despeckle(LabelResult& lr, int minArea);

/** @brief Cheap content fingerprint for cache invalidation.
 *
 * 8x8-subsampled FNV-1a over the alpha channel, quantised to 16 levels so that
 * imperceptible float noise does not invalidate the cache while a genuine change
 * always does. The thresholded bit is mixed in as well, so moving the threshold
 * across an edge counts as a change even when the alpha values themselves have
 * not moved. */
uint64_t hashAlpha(const ConstImageView& src, float threshold);

/** @brief The same fingerprint over RGB, for the custom shatter map input. */
uint64_t hashRGB(const ConstImageView& src);

}  // namespace shp
