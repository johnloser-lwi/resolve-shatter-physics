// CPU vs Metal: the same frame, both ways, compared pixel by pixel.
//
// The CUDA parity test confirms what the compiler already guarantees, because
// the CUDA kernel calls the CPU's own ShadePixel. This one does real work: the
// Metal kernel is a hand port in another language (see src/render/MetalRender.h),
// so this test is the ONLY thing holding the two in agreement. That is why it
// covers more than its CUDA twin -- the bevel, depth shading and a Fusion-style
// layout each exercise code that was ported separately and could drift alone.
//
// It runs through MetalCompositeHostSync, which flips the frames into
// bottom-up, padded-stride GPU buffers first, so the flip and the stride the
// plugin relies on are under test too, not only the shading.
//
// Skips itself, passing, when Metal is unavailable.
#include "Ballistic.h"
#include "Cells.h"
#include "Compositor.h"
#include "render/MetalRender.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void check(bool okv, const char* what) {
  std::printf("%s: %s\n", okv ? "  ok" : "FAIL", what);
  if (!okv) ++g_failures;
}

struct Frame {
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

// Deliberately not symmetric and not grey: a mirrored axis or a swapped channel
// has to show up as a difference, and a flat grey frame would hide both. The
// alpha ramp at the bar's ends puts partial coverage under test as well.
Frame makeSource(int w, int h) {
  Frame f;
  f.w = w;
  f.h = h;
  f.px.assign(std::size_t(w) * std::size_t(h) * 4, 0.0f);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float fx = float(x) / float(w);
      const float fy = float(y) / float(h);
      const float dx = fx - 0.42f, dy = fy - 0.38f;
      const bool blob = dx * dx + dy * dy < 0.055f;
      const bool bar = (y > h * 3 / 5 && y < h * 3 / 5 + h / 9 && x > w / 8 && x < w * 7 / 8);
      if (!blob && !bar) continue;

      const std::size_t p = (std::size_t(y) * std::size_t(w) + std::size_t(x)) * 4;
      const float a = blob ? 1.0f : 0.55f + 0.45f * fx;
      f.px[p + 0] = (0.15f + 0.85f * fx) * a;
      f.px[p + 1] = (0.85f - 0.55f * fy) * a;
      f.px[p + 2] = (0.30f + 0.60f * fy) * a;
      f.px[p + 3] = a;
    }
  }
  return f;
}

struct Case {
  const char* name;
  float t = 9.0f;
  bool edgeAA = true;
  bool showPieces = false;
  bool blur = false;
  float bevel = 0.0f;
  float depthSpread = 0.0f;
  shp::FrameLayout layout;  // default: destination == source
};

void runCase(const Case& c) {
  const Frame src = makeSource(480, 320);

  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 90;
  sp.buildEdgeField = c.bevel > 0.0f;
  auto geom = shp::buildGeometry(src.view(), nullptr, sp);
  if (geom->shards.empty()) {
    std::printf("FAIL: %s produced no shards\n", c.name);
    ++g_failures;
    return;
  }

  shp::FrameScale fs;
  fs.width = float(geom->width);
  fs.height = float(geom->height);
  fs.meanArea = geom->meanArea;
  fs.meanRadius = geom->meanRadius;

  shp::ForceParams fp;
  fp.f1.enabled = true;
  fp.f1.posX = 0.35f;
  fp.f1.posY = 0.4f;
  fp.f1.strength = 1.3f;
  shp::PhysicsParams pp;
  shp::RenderParams rp;
  shp::DepthParams dp;
  dp.spread = c.depthSpread;
  if (c.depthSpread > 0.0f) dp.aerial = 0.6f;

  std::vector<shp::ShardPose> poses;
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, dp, c.t, poses);

  shp::BlurParams mb;
  mb.enabled = c.blur;
  std::vector<shp::ShardDraw> draws;
  shp::BuildDrawList(*geom, poses, mb, dp, fs, draws, c.layout);

  const int dw = c.layout.dstWidth > 0 ? c.layout.dstWidth : src.w;
  const int dh = c.layout.dstHeight > 0 ? c.layout.dstHeight : src.h;
  shp::TileGrid tiles;
  shp::BuildTileGrid(draws, dw, dh, tiles);

  shp::CompositeParams cp;
  cp.edgeAntialias = c.edgeAA;
  cp.showPieces = c.showPieces;
  cp.bevel = c.bevel;
  cp.bevelWidth = 10.0f;
  cp.lightX = 0.6f;
  cp.lightY = -0.8f;

  Frame cpu;
  cpu.w = dw;
  cpu.h = dh;
  cpu.px.assign(std::size_t(dw) * std::size_t(dh) * 4, 0.0f);
  shp::Composite(cpu.mutableView(), src.view(), *geom, draws, tiles, cp);

  const bool edges = c.bevel > 0.0f && geom->edges.valid();
  if (c.bevel > 0.0f && !edges) {
    std::printf("FAIL: %s -- no edge field was built\n", c.name);
    ++g_failures;
    return;
  }

  std::vector<float> gpu(cpu.px.size(), -1.0f);
  const bool ran = shp::MetalCompositeHostSync(
      src.px.data(), src.w, src.h, gpu.data(), dw, dh, geom->labelImage.data(),
      edges ? geom->edges.dist.data() : nullptr, edges ? geom->edges.angle.data() : nullptr,
      geom->width, geom->height, draws.data(), int(draws.size()), tiles.start.data(),
      tiles.tileCount(), tiles.shardIdx.data(), int(tiles.shardIdx.size()), tiles.tilesX,
      tiles.tilesY, cp);
  if (!ran) {
    std::printf("FAIL: %s -- Metal path did not run\n", c.name);
    ++g_failures;
    return;
  }

  // Not bit-exact, deliberately: the compilers may contract multiply-adds
  // differently, and cos/sin in the bevel come from different libraries. A few
  // ulps is expected and meaningless. A real disagreement -- a wrong stencil, a
  // flipped axis, a missed shard -- is orders of magnitude larger.
  double worst = 0.0;
  std::size_t differing = 0, drawn = 0;
  for (std::size_t i = 0; i < cpu.px.size(); ++i) {
    const double d = std::fabs(double(cpu.px[i]) - double(gpu[i]));
    if (d > worst) worst = d;
    if (d > 1e-4) ++differing;
    if (i % 4 == 3 && cpu.px[i] > 0.0f) ++drawn;
  }

  char msg[256];
  std::snprintf(msg, sizeof(msg), "%s -- worst delta %.2e, %zu/%zu values differ, %zu px drawn",
                c.name, worst, differing, cpu.px.size(), drawn);
  // "drawn" guards against the vacuous pass: two blank frames agree perfectly.
  check(worst < 1e-3 && differing == 0 && drawn > 0, msg);
}

void testHash() {
  Frame f = makeSource(480, 320);
  const float thr = 0.15f;

  const uint64_t a = shp::MetalHashHostSync(f.px.data(), f.w, f.h, thr);
  const uint64_t b = shp::MetalHashHostSync(f.px.data(), f.w, f.h, thr);
  check(a != 0 && a == b, "GPU hash is stable for unchanged content");

  // The hash samples every 8th pixel of the BUFFER, which is bottom-up: top-down
  // row 159 of 320 is buffer row 160, on the grid. (200,159) is inside the blob,
  // so knocking its alpha out is a real content change.
  const std::size_t p = (std::size_t(159) * std::size_t(f.w) + 200) * 4 + 3;
  Frame g = f;
  g.px[p] = 0.0f;
  check(shp::MetalHashHostSync(g.px.data(), g.w, g.h, thr) != a,
        "GPU hash notices a real content change");

  Frame n = f;
  n.px[p] -= 1e-5f;
  check(shp::MetalHashHostSync(n.px.data(), n.w, n.h, thr) == a,
        "GPU hash ignores imperceptible float noise");

  check(shp::MetalHashHostSync(f.px.data(), f.w - 8, f.h, thr) != a,
        "GPU hash notices a resize");
}

}  // namespace

int main() {
  std::printf("metalparity\n");

  if (!shp::MetalAvailable()) {
    std::printf("  no Metal device, or the kernel did not compile - skipping\n");
    return 0;
  }

  // The rest frame first: if this disagrees, nothing downstream is worth
  // reading, because the two paths do not even agree on a copy.
  Case c{"rest frame, AA off"};
  c.t = 0.0f;
  c.edgeAA = false;
  runCase(c);

  c = Case{"rest frame, AA on"};
  c.t = 0.0f;
  runCase(c);

  c = Case{"mid shatter, AA off"};
  c.edgeAA = false;
  runCase(c);

  runCase(Case{"mid shatter, AA on"});

  c = Case{"mid shatter, show pieces"};
  c.showPieces = true;
  runCase(c);

  c = Case{"mid shatter, motion blur"};
  c.blur = true;
  runCase(c);

  c = Case{"mid shatter, bevel"};
  c.bevel = 0.8f;
  runCase(c);

  c = Case{"mid shatter, depth and aerial shading"};
  c.depthSpread = 0.6f;
  runCase(c);

  // Inside a Fusion comp the destination is not the source: here it is larger
  // and offset, so pixels outside the source must come out transparent and the
  // rest must sample the right place.
  c = Case{"Fusion layout, larger offset destination"};
  c.layout.dstWidth = 480 + 64;
  c.layout.dstHeight = 320 + 48;
  c.layout.offX = -32;
  c.layout.offY = -16;
  runCase(c);

  c = Case{"everything at once"};
  c.blur = true;
  c.bevel = 0.8f;
  c.depthSpread = 0.6f;
  c.layout.dstWidth = 480 - 40;
  c.layout.dstHeight = 320 - 24;
  c.layout.offX = 20;
  c.layout.offY = 12;
  runCase(c);

  testHash();

  if (g_failures) {
    std::printf("\n%d parity check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nCPU and Metal agree\n");
  return 0;
}
