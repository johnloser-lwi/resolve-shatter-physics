// The geometry behind dragging a rotated box's corner handles.
//
// This lives in core, away from the overlay, for one reason: it is the part
// that was wrong twice and could not be tested where it was. The overlay needs
// OFX headers, so nothing in it can be reached from the test suite; these are
// plain functions over Vec2 in whatever 2D frame the caller is using, and the
// tests exercise them directly.
//
// The two mistakes both worth remembering:
//
//   1. A box stores its corners UNROTATED and rotates them only when drawing.
//      The handles must therefore be drawn rotated too, or they sit off the
//      outline they are supposed to belong to.
//
//   2. Resizing must pivot on something that does not move. The obvious
//      implementation un-rotates the pointer about the box's centre -- but the
//      centre is the midpoint of the two corners, one of which the drag is
//      rewriting, so every event is measured against a pivot the previous event
//      already shifted. The result compounds and the handle becomes impossible
//      to control. Anchoring the opposite corner instead makes the gesture
//      exact and stable.
#pragma once

#include "TransformMath.h"

namespace shp {

/** @brief Turn @p p about @p pivot by @p radians. */
SHP_HD inline Vec2 RotateAbout(const Vec2& p, const Vec2& pivot, float radians) {
  const float ca = cosf(radians);
  const float sa = sinf(radians);
  const float dx = p.x - pivot.x;
  const float dy = p.y - pivot.y;
  return Vec2(pivot.x + dx * ca - dy * sa, pivot.y + dx * sa + dy * ca);
}

SHP_HD inline Vec2 MidPoint(const Vec2& a, const Vec2& b) {
  return Vec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
}

/** @brief Where a stored corner is actually drawn.
 *
 * @param p1,p2   the box's two stored corners, unrotated
 * @param radians rotation applied about the centre when drawing
 */
SHP_HD inline Vec2 DrawnCorner(const Vec2& stored, const Vec2& p1, const Vec2& p2,
                               float radians) {
  return RotateAbout(stored, MidPoint(p1, p2), radians);
}

/** @brief Resize a rotated box by dragging one drawn corner.
 *
 * @param anchorDrawn  the OTHER drawn corner, captured when the drag began and
 *                     held fixed for its duration
 * @param pointerDrawn where the pointer is now
 * @param radians      the same rotation DrawnCorner would apply
 * @param outDragged   new stored value for the corner being dragged
 * @param outFixed     new stored value for the corner being held
 *
 * Opposite corners of a box share the centre as their midpoint, and rotation
 * preserves that -- so the new centre follows from the anchor and the pointer
 * alone, with no dependence on the values being rewritten. That is what makes
 * this stable under repeated application.
 */
SHP_HD inline void ResizeRotatedBox(const Vec2& anchorDrawn, const Vec2& pointerDrawn,
                                    float radians, Vec2& outDragged, Vec2& outFixed) {
  const Vec2 mid = MidPoint(anchorDrawn, pointerDrawn);
  outDragged = RotateAbout(pointerDrawn, mid, -radians);
  outFixed = RotateAbout(anchorDrawn, mid, -radians);
}

}  // namespace shp
