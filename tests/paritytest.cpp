// CPU vs GPU: the same frame, both ways, compared pixel by pixel.
//
// The two paths call the identical ShadePixel through the SHP_HD macro, so in
// principle they cannot disagree. This exists because "in principle" is exactly
// what a parity test is for: a stray `-use_fast_math`, a different rounding mode
// on a transcendental, an uninitialised device buffer, or a launch that silently
// failed and left the destination untouched all produce a plausible picture and
// none of them are visible by eye.
//
// Skips itself, passing, when no CUDA device is present -- the suite has to work
// on a machine without a GPU.
#include "Ballistic.h"
#include "Cells.h"
#include "Compositor.h"
#include "render/CudaRender.h"

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
// has to show up as a difference, and a flat grey frame would hide both.
Frame makeSource(int w, int h) {
  Frame f;
  f.w = w;
  f.h = h;
  f.px.assign(std::size_t(w) * std::size_t(h) * 4, 0.0f);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float fx = float(x) / float(w);
      const float fy = float(y) / float(h);
      // An off-centre blob plus a couple of bars, so there are real edges.
      const float dx = fx - 0.42f, dy = fy - 0.38f;
      const bool blob = dx * dx + dy * dy < 0.055f;
      const bool bar = (y > h * 3 / 5 && y < h * 3 / 5 + h / 9 && x > w / 8 && x < w * 7 / 8);
      if (!blob && !bar) continue;

      const std::size_t p = (std::size_t(y) * std::size_t(w) + std::size_t(x)) * 4;
      const float a = 1.0f;
      f.px[p + 0] = (0.15f + 0.85f * fx) * a;
      f.px[p + 1] = (0.85f - 0.55f * fy) * a;
      f.px[p + 2] = (0.30f + 0.60f * fy) * a;
      f.px[p + 3] = a;
    }
  }
  return f;
}

bool runCase(const char* name, bool edgeAA, bool showPieces, bool blur, float t) {
  const Frame src = makeSource(480, 320);

  shp::ShapeParams sp;
  sp.pattern = shp::Pattern::Voronoi;
  sp.cellCount = 90;
  auto geom = shp::buildGeometry(src.view(), nullptr, sp);
  if (geom->shards.empty()) {
    std::printf("FAIL: %s produced no shards\n", name);
    ++g_failures;
    return false;
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

  std::vector<shp::ShardPose> poses;
  shp::DepthParams dp;
  shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, dp, t, poses);

  shp::BlurParams mb;
  mb.enabled = blur;
  std::vector<shp::ShardDraw> draws;
  shp::BuildDrawList(*geom, poses, mb, dp, fs, draws);

  shp::TileGrid tiles;
  shp::BuildTileGrid(draws, geom->width, geom->height, tiles);

  shp::CompositeParams cp;
  cp.edgeAntialias = edgeAA;
  cp.showPieces = showPieces;

  Frame cpu;
  cpu.w = src.w;
  cpu.h = src.h;
  cpu.px.assign(src.px.size(), 0.0f);
  shp::Composite(cpu.mutableView(), src.view(), *geom, draws, tiles, cp);

  Frame gpu;
  gpu.w = src.w;
  gpu.h = src.h;
  gpu.px.assign(src.px.size(), -1.0f);  // poison, so an untouched buffer fails loudly

  const bool ran = shp::CudaCompositeHostSync(
      src.px.data(), gpu.px.data(), geom->labelImage.data(), nullptr, nullptr, src.w, src.h, draws.data(),
      int(draws.size()), tiles.start.data(), tiles.tileCount(), tiles.shardIdx.data(),
      int(tiles.shardIdx.size()), tiles.tilesX, tiles.tilesY, cp);
  if (!ran) {
    std::printf("FAIL: %s -- CUDA path did not run\n", name);
    ++g_failures;
    return false;
  }

  // Not bit-exact, deliberately. Both paths run the same arithmetic but the
  // compilers are free to contract multiply-adds differently, so a few ulps of
  // difference is expected and meaningless. A real disagreement -- a wrong
  // stencil, a flipped axis, a missed shard -- is orders of magnitude larger.
  double worst = 0.0;
  std::size_t differing = 0;
  for (std::size_t i = 0; i < cpu.px.size(); ++i) {
    const double d = std::fabs(double(cpu.px[i]) - double(gpu.px[i]));
    if (d > worst) worst = d;
    if (d > 1e-4) ++differing;
  }

  char msg[256];
  std::snprintf(msg, sizeof(msg), "%s -- worst delta %.2e, %zu/%zu pixels differ", name, worst,
                differing / 4, cpu.px.size() / 4);
  check(worst < 1e-3 && differing == 0, msg);
  return true;
}

}  // namespace

int main() {
  std::printf("paritytest\n");

  if (!shp::CudaAvailable()) {
    std::printf("  no CUDA device present - skipping (this is not a failure)\n");
    return 0;
  }

  // The rest frame first: if this disagrees, nothing downstream is worth
  // reading, because it means the two paths do not even agree on a copy.
  runCase("rest frame, AA off", false, false, false, 0.0f);
  runCase("rest frame, AA on", true, false, false, 0.0f);
  runCase("mid shatter, AA off", false, false, false, 9.0f);
  runCase("mid shatter, AA on", true, false, false, 9.0f);
  runCase("mid shatter, show pieces", true, true, false, 9.0f);
  runCase("mid shatter, motion blur", true, false, true, 9.0f);

  if (g_failures) {
    std::printf("\n%d parity check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nCPU and GPU agree\n");
  return 0;
}
