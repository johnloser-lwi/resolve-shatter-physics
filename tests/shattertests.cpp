// The properties that are expensive to discover are broken.
//
// Every check here exists because getting it wrong produces something that still
// renders a plausible-looking picture -- which is why they are assertions about
// counts, orders and exact reproduction rather than about how a frame looks.
//
// Run: ctest --test-dir build --output-on-failure
#include "Ballistic.h"
#include "Cells.h"
#include "Compositor.h"
#include "Physics2D.h"
#include "Segmentation.h"
#include "SimCache.h"
#include "SliceList.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  } else {
    std::printf("  ok: %s\n", what);
  }
}

// A synthetic frame: three opaque discs on transparent black, deliberately of
// different sizes and NOT symmetric, so a flipped axis or a swapped width and
// height cannot pass unnoticed.
struct TestImage {
  int w = 0, h = 0;
  std::vector<float> px;

  shp::ConstImageView view() const {
    shp::ConstImageView v;
    v.data = px.data();
    v.width = w;
    v.height = h;
    v.rowStride = std::ptrdiff_t(w) * 4;
    return v;
  }
  shp::ImageView mutableView() {
    shp::ImageView v;
    v.data = px.data();
    v.width = w;
    v.height = h;
    v.rowStride = std::ptrdiff_t(w) * 4;
    return v;
  }
};

TestImage makeDiscs() {
  TestImage img;
  img.w = 320;
  img.h = 200;
  img.px.assign(std::size_t(img.w) * std::size_t(img.h) * 4, 0.0f);

  struct Disc { float cx, cy, r; };
  const Disc discs[3] = {{60.0f, 60.0f, 34.0f}, {160.0f, 90.0f, 22.0f}, {250.0f, 140.0f, 44.0f}};

  for (int y = 0; y < img.h; ++y) {
    for (int x = 0; x < img.w; ++x) {
      for (const Disc& d : discs) {
        const float dx = float(x) + 0.5f - d.cx;
        const float dy = float(y) + 0.5f - d.cy;
        if (dx * dx + dy * dy > d.r * d.r) continue;
        const std::size_t p = (std::size_t(y) * std::size_t(img.w) + std::size_t(x)) * 4;
        img.px[p + 0] = 0.8f;
        img.px[p + 1] = 0.6f;
        img.px[p + 2] = 0.4f;
        img.px[p + 3] = 1.0f;
        break;
      }
    }
  }
  return img;
}

shp::FrameScale scaleOf(const shp::ShatterGeometry& g) {
  shp::FrameScale fs;
  fs.width = float(g.width);
  fs.height = float(g.height);
  fs.meanArea = g.meanArea;
  fs.meanRadius = g.meanRadius;
  return fs;
}

bool posesIdentical(const std::vector<shp::ShardPose>& a, const std::vector<shp::ShardPose>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    // Bit-exact, not approximate. "Close enough" would hide exactly the kind of
    // drift that makes a cached simulation disagree with a replayed one.
    if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].angle != b[i].angle ||
        a[i].scale != b[i].scale || a[i].alpha != b[i].alpha)
      return false;
  }
  return true;
}

// --------------------------------------------------------------------- tests

// Alpha mode must find exactly the three discs. If the labeller's connectivity
// or its despeckle threshold drifts, this is the first thing to notice.
void testAlphaPieceCount() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Alpha;

  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  check(geom->shards.size() == 3, "alpha mode finds exactly the three discs");

  // And each piece's centroid must land inside the frame, which catches a
  // centroid accumulated in the wrong coordinate space.
  bool inside = true;
  for (const shp::Shard& s : geom->shards)
    if (s.cx < 0 || s.cy < 0 || s.cx >= float(img.w) || s.cy >= float(img.h)) inside = false;
  check(inside, "piece centroids lie inside the frame");
}

// Cells only exist where the image does. A pattern that ignored alpha would
// return roughly cellCount pieces covering the whole frame.
void testPatternRespectsAlpha() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 200;

  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  check(!geom->shards.empty(), "voronoi produces pieces");

  std::size_t covered = 0;
  for (int32_t l : geom->labelImage)
    if (l != 0) ++covered;

  // The discs cover about 12% of this frame. Anything near 100% means the
  // pattern was not intersected with the alpha at all.
  const double frac = double(covered) / double(geom->labelImage.size());
  check(frac > 0.05 && frac < 0.30, "labelled area matches the discs, not the frame");
}

// Every label in the image must index a real shard. A despeckle that dropped
// pieces without renumbering would leave holes here, and the compositor indexes
// shards by label - 1.
void testLabelsAreDense() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 120;
  sp.minPieceArea = 0.002f;  // aggressive, to force renumbering

  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  const int n = int(geom->shards.size());

  bool ok = true;
  for (int32_t l : geom->labelImage)
    if (l < 0 || l > n) ok = false;
  check(ok, "no label points past the end of the shard list");

  bool idsMatch = true;
  for (int i = 0; i < n; ++i)
    if (geom->shards[std::size_t(i)].id != i + 1) idsMatch = false;
  check(idsMatch, "shard ids are consecutive from 1");
}

// Collision proxies must be usable polygons: convex, wound consistently, and
// with a positive area. A hull wound the wrong way makes every contact normal
// point inward, and pieces suck together instead of bouncing.
void testHullsAreValid() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Alpha;
  sp.buildPolygons = true;

  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  bool ok = !geom->shards.empty();
  for (const shp::Shard& s : geom->shards) {
    if (s.poly.size() < 3) { ok = false; continue; }
    if (shp::PolygonArea(s.poly) <= 0.0f) ok = false;   // winding
    if (s.inertiaPerMass <= 0.0f) ok = false;
    if (s.radius <= 0.0f) ok = false;
  }
  check(ok, "hulls are non-degenerate, positively wound, with real inertia");
}

// The stateless mode's whole justification is that it can be evaluated in any
// order. So it must not matter what order it is evaluated in.
void testBallisticDeterminism() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 80;
  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  const shp::FrameScale fs = scaleOf(*geom);

  shp::ForceParams fp;
  fp.f1.enabled = true;
  shp::PhysicsParams pp;
  shp::RenderParams rp;

  std::vector<shp::ShardPose> a, b;
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 17.0f, a);
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 17.0f, b);
  check(posesIdentical(a, b), "ballistic: same inputs give bit-identical poses");

  // Evaluating an earlier frame in between must change nothing about frame 17.
  std::vector<shp::ShardPose> mid;
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 3.0f, mid);
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 17.0f, b);
  check(posesIdentical(a, b), "ballistic: evaluation order does not matter");

  // A different motion seed must actually change the motion.
  fp.seed = 99;
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 17.0f, b);
  check(!posesIdentical(a, b), "ballistic: the motion seed changes the motion");
}

// The stateful mode's correctness rests entirely on replaying identically. If
// this fails, scrubbing backward shows a different shatter than playing forward.
void testRigidDeterminism() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 60;
  sp.buildPolygons = true;
  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  const shp::FrameScale fs = scaleOf(*geom);

  shp::ForceParams fp;
  fp.f1.enabled = true;
  shp::PhysicsParams pp;
  pp.floorEnabled = true;
  shp::RenderParams rp;

  // Forward, straight to frame 40.
  shp::SimCache c1;
  std::vector<shp::ShardPose> straight;
  c1.getPoses(geom, fp, pp, rp, fs, 40.0f, straight);

  // A separate cache walked one frame at a time to the same place.
  shp::SimCache c2;
  std::vector<shp::ShardPose> walked;
  for (int f = 1; f <= 40; ++f) c2.getPoses(geom, fp, pp, rp, fs, float(f), walked);
  check(posesIdentical(straight, walked), "rigid: stepping there equals jumping there");

  // Scrub backward then forward again on the SAME cache: the backward lookup
  // must not disturb the stored timeline.
  std::vector<shp::ShardPose> back, again;
  c1.getPoses(geom, fp, pp, rp, fs, 5.0f, back);
  c1.getPoses(geom, fp, pp, rp, fs, 40.0f, again);
  check(posesIdentical(straight, again), "rigid: scrubbing backward does not corrupt the run");

  // A fresh cache with the same inputs reproduces the run exactly.
  shp::SimCache c3;
  std::vector<shp::ShardPose> fresh;
  c3.getPoses(geom, fp, pp, rp, fs, 40.0f, fresh);
  check(posesIdentical(straight, fresh), "rigid: a rebuilt cache reproduces the run");
}

// A directional force pushes every piece the SAME way, unlike a radial one
// which pushes each piece away from a point. If the two ever collapse into the
// same behaviour, this is what notices.
void testDirectionalForce() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 80;
  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  const shp::FrameScale fs = scaleOf(*geom);

  shp::PhysicsParams pp;
  pp.gravity = 0.0f;      // isolate the force
  pp.randomness = 0.0f;   // and its direction
  shp::RenderParams rp;

  shp::ForceParams fp;
  fp.f1.enabled = true;
  fp.f1.shape = shp::ForceShape::Directional;
  fp.f1.direction = 90.0f;  // to the right
  fp.f1.width = 4.0f;       // wide enough to cover the whole frame
  fp.f1.strength = 1.0f;

  std::vector<shp::ShardPose> poses;
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 10.0f, poses);

  bool allRight = true;
  int moved = 0;
  for (const shp::ShardPose& p : poses) {
    if (!p.active) continue;
    if (p.x <= 0.0f) allRight = false;
    if (std::fabs(p.y) > std::fabs(p.x) * 0.05f) allRight = false;  // essentially no drift
    ++moved;
  }
  check(moved > 0 && allRight, "directional force moves every piece the same way");

  // A narrow band must leave pieces outside it alone. Aimed across the frame
  // through the middle, the discs at top-left and bottom-right fall outside.
  fp.f1.width = 0.06f;
  fp.f1.posY = 0.5f;
  fp.mode = shp::ForceMode::InsideRadius;  // so untouched pieces never detach
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 10.0f, poses);

  int active = 0;
  for (const shp::ShardPose& p : poses)
    if (p.active) ++active;
  check(active > 0 && active < int(poses.size()),
        "a narrow band moves some pieces and leaves the rest standing");
}

// A delayed force must act on pieces that are ALREADY MOVING, not be averaged
// into how they left. The two are easy to confuse and look completely different.
void testForceDelay() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 60;
  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  const shp::FrameScale fs = scaleOf(*geom);

  shp::PhysicsParams pp;
  pp.gravity = 0.0f;
  pp.randomness = 0.0f;
  pp.viscosity = 0.0f;
  shp::RenderParams rp;

  shp::ForceParams fp;
  fp.f1.enabled = true;
  fp.f1.shape = shp::ForceShape::Directional;
  fp.f1.direction = 90.0f;
  fp.f1.width = 4.0f;
  fp.f1.strength = 1.0f;
  fp.f1.delay = 10.0f;

  std::vector<shp::ShardPose> before, after;
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 5.0f, before);
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 15.0f, after);

  // Nothing has been pushed yet at frame 5 -- with gravity off, nothing has
  // moved at all, even though the pieces have detached.
  bool stillAtRest = true;
  for (const shp::ShardPose& p : before)
    if (std::fabs(p.x) > 1e-3f || std::fabs(p.y) > 1e-3f) stillAtRest = false;
  check(stillAtRest, "a delayed force has not fired yet before its delay");

  bool movedAfter = false;
  for (const shp::ShardPose& p : after)
    if (p.x > 1.0f) movedAfter = true;
  check(movedAfter, "a delayed force fires once its delay has passed");

  // The second force must add to the motion of a piece already in flight, so
  // arriving later still produces MORE displacement than never arriving.
  shp::ForceParams two = fp;
  two.f1.delay = 0.0f;
  two.f2 = two.f1;
  two.f2.delay = 8.0f;

  std::vector<shp::ShardPose> one, both;
  shp::ForceParams justOne = two;
  justOne.f2.enabled = false;
  shp::EvaluateBallistic(geom->shards, justOne, pp, rp, fs, 20.0f, one);
  shp::EvaluateBallistic(geom->shards, two, pp, rp, fs, 20.0f, both);

  double sumOne = 0.0, sumBoth = 0.0;
  for (std::size_t i = 0; i < one.size(); ++i) {
    sumOne += double(one[i].x);
    sumBoth += double(both[i].x);
  }
  check(sumBoth > sumOne * 1.2, "a delayed second force adds to a piece already in flight");

  // And the rigid-body path has to agree that a delay is a delay.
  //
  // Inside Radius Only deliberately, so the pieces stay immovable until the
  // force actually arrives. Under All At Once they detach on the start frame
  // and adjacent hulls immediately push each other apart -- correct physics,
  // but it means "has not moved" is not the thing being measured.
  sp.buildPolygons = true;
  auto rgeom = shp::buildGeometry(img.view(), nullptr, sp);
  const shp::FrameScale rfs = scaleOf(*rgeom);

  shp::ForceParams rfp = fp;
  rfp.mode = shp::ForceMode::InsideRadius;

  shp::SimCache sim;
  std::vector<shp::ShardPose> rb, ra;
  sim.getPoses(rgeom, rfp, pp, rp, rfs, 5.0f, rb);
  sim.getPoses(rgeom, rfp, pp, rp, rfs, 15.0f, ra);

  double moveBefore = 0.0, moveAfter = 0.0;
  int activeBefore = 0;
  for (const shp::ShardPose& p : rb) {
    moveBefore += std::fabs(double(p.x));
    if (p.active) ++activeBefore;
  }
  for (const shp::ShardPose& p : ra) moveAfter += std::fabs(double(p.x));

  char buf[160];
  std::snprintf(buf, sizeof(buf),
                "rigid body honours the force delay (active@5=%d, move@5=%.2f, move@15=%.1f)",
                activeBefore, moveBefore, moveAfter);
  check(activeBefore == 0 && moveBefore < 1.0 && moveAfter > 10.0, buf);
}

// A cut that fully crosses a piece separates it; one that stops halfway does
// not. That asymmetry is the whole behaviour of the slice tool.
void testManualSlices() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Manual;

  auto base = shp::buildGeometry(img.view(), nullptr, sp);
  check(base->shards.size() == 3, "manual mode starts with whole objects");

  // The first disc is centred near (60,60) with radius 34 in a 320x200 frame.
  // A horizontal cut right across it should halve it and leave the others alone.
  shp::Slice through;
  through.x1 = 0.02f;
  through.y1 = 60.0f / 200.0f;
  through.x2 = 0.40f;
  through.y2 = 60.0f / 200.0f;

  sp.slices = {through};
  auto cut = shp::buildGeometry(img.view(), nullptr, sp);
  check(cut->shards.size() == 4, "a cut across a piece splits it in two");

  // Nothing may be lost to the cut. The barrier pixels have to be handed back
  // to a neighbour, or every slice would erase a hairline of the artwork.
  std::size_t beforeCovered = 0, afterCovered = 0;
  for (int32_t l : base->labelImage)
    if (l != 0) ++beforeCovered;
  for (int32_t l : cut->labelImage)
    if (l != 0) ++afterCovered;
  check(beforeCovered == afterCovered, "a cut separates without erasing any pixels");

  // A stub that stops inside the disc must not separate anything.
  shp::Slice stub;
  stub.x1 = 0.02f;
  stub.y1 = 60.0f / 200.0f;
  stub.x2 = 0.10f;  // ends well short of the far edge
  stub.y2 = 60.0f / 200.0f;

  sp.slices = {stub};
  auto partial = shp::buildGeometry(img.view(), nullptr, sp);
  check(partial->shards.size() == 3, "a cut that stops halfway separates nothing");

  // Cuts apply on top of a generated pattern, not instead of it.
  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 40;
  sp.slices.clear();
  auto plain = shp::buildGeometry(img.view(), nullptr, sp);
  sp.slices = {through};
  auto sliced = shp::buildGeometry(img.view(), nullptr, sp);
  check(sliced->shards.size() > plain->shards.size(),
        "cuts apply on top of a generated pattern");

  // Diagonal cuts are the case a one-pixel barrier leaks through, because an
  // 8-connected labeller steps diagonally too.
  shp::Slice diag;
  diag.x1 = 0.02f;
  diag.y1 = 0.05f;
  diag.x2 = 0.40f;
  diag.y2 = 0.55f;
  sp.pattern = shp::Pattern::Manual;
  sp.slices = {diag};
  auto diagonal = shp::buildGeometry(img.view(), nullptr, sp);
  check(diagonal->shards.size() > 3, "a diagonal cut does not leak");
}

// The cut list is persisted as text, so it has to survive a round trip and it
// has to tolerate being edited by hand.
void testSliceListRoundTrip() {
  std::vector<shp::Slice> in;
  shp::Slice a;
  a.x1 = 0.125f; a.y1 = 0.25f; a.x2 = 0.875f; a.y2 = 0.75f;
  shp::Slice b;
  b.x1 = -0.5f; b.y1 = 0.0f; b.x2 = 1.5f; b.y2 = 1.0f;
  in.push_back(a);
  in.push_back(b);

  const std::vector<shp::Slice> out = shp::ParseSlices(shp::FormatSlices(in));
  bool same = out.size() == in.size();
  for (std::size_t i = 0; same && i < in.size(); ++i) {
    if (std::fabs(out[i].x1 - in[i].x1) > 1e-4f) same = false;
    if (std::fabs(out[i].y2 - in[i].y2) > 1e-4f) same = false;
  }
  check(same, "the cut list survives a write/read round trip");

  // Half-typed input must not throw or invent a cut from a partial group.
  check(shp::ParseSlices("0.1,0.2,0.3").empty(), "a partial cut is dropped, not guessed");
  check(shp::ParseSlices("").empty(), "an empty cut list parses to nothing");
  check(shp::ParseSlices("garbage").empty(), "unparseable text yields no cuts");
  check(shp::ParseSlices("0.1 0.2 0.3 0.4  0.5 0.6 0.7 0.8").size() == 2,
        "separators are lenient");
}

// Changing a render-only control must NOT invalidate the simulation, or every
// fade tweak would re-run the physics.
void testFadeDoesNotInvalidateSim() {
  shp::ForceParams fp;
  shp::PhysicsParams pp;
  const uint64_t before = shp::HashSimParams(fp, pp);

  shp::RenderParams rp;
  rp.fadeDuration = 24.0f;
  rp.shrinkOnFade = true;
  (void)rp;  // render params are deliberately not part of the key

  const uint64_t after = shp::HashSimParams(fp, pp);
  check(before == after, "sim key ignores render-only controls");

  pp.gravity += 0.5f;
  check(shp::HashSimParams(fp, pp) != before, "sim key notices a gravity change");
}

// Pieces must land on the floor and stay there, and must not sink through it.
// A flipped contact normal passes every other test in this file and fails here.
void testPiecesRestOnFloor() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Alpha;
  sp.buildPolygons = true;
  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  const shp::FrameScale fs = scaleOf(*geom);

  shp::ForceParams fp;
  fp.f1.enabled = true;
  fp.f1.strength = 0.0f;   // no blast: just let them fall
  shp::PhysicsParams pp;
  pp.gravity = 6.0f;
  pp.floorEnabled = true;
  pp.floorY = 1.0f;
  pp.bounciness = 0.0f;
  shp::RenderParams rp;

  shp::SimCache sim;
  std::vector<shp::ShardPose> poses;
  sim.getPoses(geom, fp, pp, rp, fs, 200.0f, poses);

  const float floorY = fs.height;
  bool allAbove = true;
  bool allSettled = true;
  for (std::size_t i = 0; i < poses.size(); ++i) {
    const float cy = geom->shards[i].cy + poses[i].y;
    // Allowed a shard's radius of slack: the centroid sits above the floor by
    // roughly that much when the piece is resting on it.
    if (cy > floorY + geom->shards[i].radius + 4.0f) allAbove = false;
    if (std::fabs(poses[i].vy) > 2.0f) allSettled = false;
  }
  check(allAbove, "no piece sinks through the floor");
  check(allSettled, "pieces come to rest rather than falling forever");
}

// Before the start frame nothing has moved, so the output has to be the input.
// This is the check that catches a sign error in the view flip or a half-pixel
// drift in the sampler.
void testRestFrameReproducesSource() {
  TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 60;
  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  const shp::FrameScale fs = scaleOf(*geom);

  shp::ForceParams fp;
  fp.f1.enabled = true;
  fp.startFrame = 10;      // shatter has not begun at t = 0
  shp::PhysicsParams pp;
  shp::RenderParams rp;

  std::vector<shp::ShardPose> poses;
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 0.0f, poses);

  std::vector<shp::ShardDraw> draws;
  shp::BuildDrawList(*geom, poses, shp::BlurParams{}, draws);
  shp::TileGrid tiles;
  shp::BuildTileGrid(draws, geom->width, geom->height, tiles);

  TestImage out;
  out.w = img.w;
  out.h = img.h;
  out.px.assign(img.px.size(), 0.0f);

  shp::CompositeParams cp;
  cp.edgeAntialias = false;   // exact stencil, so this can be a tight compare
  shp::Composite(out.mutableView(), img.view(), *geom, draws, tiles, cp);

  // Every pixel the shatter claims must come back with its original colour.
  double worst = 0.0;
  std::size_t claimed = 0;
  for (std::size_t p = 0; p < geom->labelImage.size(); ++p) {
    if (geom->labelImage[p] == 0) continue;
    ++claimed;
    for (int c = 0; c < 4; ++c) {
      const double d = std::fabs(double(out.px[p * 4 + std::size_t(c)]) -
                                 double(img.px[p * 4 + std::size_t(c)]));
      if (d > worst) worst = d;
    }
  }
  check(claimed > 0, "the rest frame claims some pixels");
  check(worst < 1e-4, "the rest frame reproduces the source exactly");

  // The same frame with motion blur switched on. A resting piece has no
  // velocity, so it must come back byte-identical -- and if the shutter samples
  // were composited OVER one another instead of averaged, a solid piece would
  // come out around 63% opaque and this would catch it.
  shp::BlurParams mb;
  mb.enabled = true;
  mb.samples = 8;

  std::vector<shp::ShardDraw> bdraws;
  shp::BuildDrawList(*geom, poses, mb, bdraws);
  shp::TileGrid btiles;
  shp::BuildTileGrid(bdraws, geom->width, geom->height, btiles);

  TestImage bout;
  bout.w = img.w;
  bout.h = img.h;
  bout.px.assign(img.px.size(), 0.0f);
  shp::Composite(bout.mutableView(), img.view(), *geom, bdraws, btiles, cp);

  double bworst = 0.0;
  for (std::size_t p = 0; p < geom->labelImage.size(); ++p) {
    if (geom->labelImage[p] == 0) continue;
    for (int c = 0; c < 4; ++c)
      bworst = std::max(bworst, std::fabs(double(bout.px[p * 4 + std::size_t(c)]) -
                                          double(img.px[p * 4 + std::size_t(c)])));
  }
  check(bworst < 1e-4, "motion blur leaves a resting piece fully solid");
}

// A moving piece under blur must spread across more pixels than it does without,
// and must not gain or lose energy doing it.
void testMotionBlurSpreads() {
  const TestImage img = makeDiscs();
  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Alpha;
  auto geom = shp::buildGeometry(img.view(), nullptr, sp);
  const shp::FrameScale fs = scaleOf(*geom);

  shp::ForceParams fp;
  fp.f1.enabled = true;
  fp.f1.strength = 3.0f;
  shp::PhysicsParams pp;
  shp::RenderParams rp;

  std::vector<shp::ShardPose> poses;
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, 6.0f, poses);

  auto renderWith = [&](const shp::BlurParams& mb, TestImage& out) {
    std::vector<shp::ShardDraw> draws;
    shp::BuildDrawList(*geom, poses, mb, draws);
    shp::TileGrid tiles;
    shp::BuildTileGrid(draws, geom->width, geom->height, tiles);
    out.w = img.w;
    out.h = img.h;
    out.px.assign(img.px.size(), 0.0f);
    shp::CompositeParams cp;
    shp::Composite(out.mutableView(), img.view(), *geom, draws, tiles, cp);
  };

  TestImage sharp, blurred;
  renderWith(shp::BlurParams{}, sharp);

  shp::BlurParams mb;
  mb.enabled = true;
  renderWith(mb, blurred);

  auto touched = [](const TestImage& t) {
    std::size_t n = 0;
    for (std::size_t p = 3; p < t.px.size(); p += 4)
      if (t.px[p] > 0.004f) ++n;
    return n;
  };
  auto totalAlpha = [](const TestImage& t) {
    double s = 0.0;
    for (std::size_t p = 3; p < t.px.size(); p += 4) s += double(t.px[p]);
    return s;
  };

  check(touched(blurred) > touched(sharp), "blur spreads a moving piece over more pixels");

  // Averaging conserves the total; over-compositing would lose a chunk of it.
  const double a = totalAlpha(sharp), b = totalAlpha(blurred);
  check(std::fabs(a - b) / std::max(1.0, a) < 0.10, "blur conserves total coverage");
}

// A proxy render must break into the same pieces as a full-resolution one.
// Anything measured in pixels rather than as a fraction of the frame fails here.
void testResolutionIndependence() {
  const TestImage full = makeDiscs();

  // Half-resolution copy of the same content.
  TestImage half;
  half.w = full.w / 2;
  half.h = full.h / 2;
  half.px.assign(std::size_t(half.w) * std::size_t(half.h) * 4, 0.0f);
  for (int y = 0; y < half.h; ++y)
    for (int x = 0; x < half.w; ++x)
      for (int c = 0; c < 4; ++c)
        half.px[(std::size_t(y) * std::size_t(half.w) + std::size_t(x)) * 4 + std::size_t(c)] =
            full.px[(std::size_t(y * 2) * std::size_t(full.w) + std::size_t(x * 2)) * 4 +
                    std::size_t(c)];

  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Alpha;

  auto a = shp::buildGeometry(full.view(), nullptr, sp);
  auto b = shp::buildGeometry(half.view(), nullptr, sp);
  check(a->shards.size() == b->shards.size(),
        "alpha mode finds the same piece count at half resolution");

  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 100;
  auto c = shp::buildGeometry(full.view(), nullptr, sp);
  auto d = shp::buildGeometry(half.view(), nullptr, sp);
  // Not exact -- despeckling and rasterisation differ slightly -- but a
  // pixel-denominated control would change this by a factor, not a few percent.
  const double ratio = double(d->shards.size()) / double(c->shards.size() > 0 ? c->shards.size() : 1);
  check(ratio > 0.75 && ratio < 1.25, "voronoi piece count survives a half-res render");
}

// The content fingerprint has to notice a real change and ignore float noise,
// or the analysis cache either never updates or never hits.
void testHashBehaviour() {
  TestImage img = makeDiscs();
  const uint64_t base = shp::hashAlpha(img.view(), 0.15f);
  check(base == shp::hashAlpha(img.view(), 0.15f), "hash is stable for unchanged content");

  // A change well below the quantisation step must not bust the cache.
  for (std::size_t p = 3; p < img.px.size(); p += 4)
    if (img.px[p] > 0.5f) img.px[p] = 0.999f;
  check(base == shp::hashAlpha(img.view(), 0.15f), "hash ignores imperceptible float noise");

  // Erasing a disc must.
  for (int y = 0; y < 80; ++y)
    for (int x = 0; x < 110; ++x)
      img.px[(std::size_t(y) * std::size_t(img.w) + std::size_t(x)) * 4 + 3] = 0.0f;
  check(base != shp::hashAlpha(img.view(), 0.15f), "hash notices a real content change");
}

}  // namespace

int main() {
  std::printf("shattertests\n");
  testAlphaPieceCount();
  testPatternRespectsAlpha();
  testLabelsAreDense();
  testHullsAreValid();
  testBallisticDeterminism();
  testRigidDeterminism();
  testDirectionalForce();
  testForceDelay();
  testManualSlices();
  testSliceListRoundTrip();
  testFadeDoesNotInvalidateSim();
  testPiecesRestOnFloor();
  testRestFrameReproducesSource();
  testMotionBlurSpreads();
  testResolutionIndependence();
  testHashBehaviour();

  if (g_failures) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nall checks passed\n");
  return 0;
}
