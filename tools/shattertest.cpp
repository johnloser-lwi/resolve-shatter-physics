// Offline harness: PNG in, shattered frames out.
//
// This links the *same* src/core the plugin does, which is the whole point of
// the no-OFX-headers rule. Tuning a shatter here costs a second; tuning it
// through Resolve costs a relaunch, and a relaunch discourages exactly the kind
// of iteration a physics effect needs.
//
//   shattertest in.png out.png                       one frame of showPieces tint
//   shattertest in.png out.png --pattern voronoi --cells 80
//   shattertest in.png out.png --frames 24 --strip 8 contact sheet of a shatter
//
// The CPU compositor is what runs here; the CUDA kernel is a separate
// implementation, so a good-looking harness render is not proof the GPU path
// agrees. That is what the parity test is for.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Ballistic.h"
#include "Cells.h"
#include "Compositor.h"
#include "Hash.h"
#include "ImageView.h"
#include "SimCache.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Bitmap {
  int w = 0, h = 0;
  std::vector<float> px;  // RGBA, premultiplied, top-down

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

bool loadPng(const char* path, Bitmap& out) {
  int n = 0;
  unsigned char* data = stbi_load(path, &out.w, &out.h, &n, 4);
  if (!data) {
    std::fprintf(stderr, "cannot read %s\n", path);
    return false;
  }
  out.px.resize(std::size_t(out.w) * std::size_t(out.h) * 4);
  for (std::size_t i = 0; i < std::size_t(out.w) * std::size_t(out.h); ++i) {
    const float a = float(data[i * 4 + 3]) / 255.0f;
    // Premultiply on load: that is the space Resolve hands plugins, so the
    // harness has to work in it too or the compositing will not match.
    out.px[i * 4 + 0] = float(data[i * 4 + 0]) / 255.0f * a;
    out.px[i * 4 + 1] = float(data[i * 4 + 1]) / 255.0f * a;
    out.px[i * 4 + 2] = float(data[i * 4 + 2]) / 255.0f * a;
    out.px[i * 4 + 3] = a;
  }
  stbi_image_free(data);
  return true;
}

bool savePng(const char* path, const Bitmap& b) {
  std::vector<unsigned char> out(std::size_t(b.w) * std::size_t(b.h) * 4);
  for (std::size_t i = 0; i < std::size_t(b.w) * std::size_t(b.h); ++i) {
    const float a = b.px[i * 4 + 3];
    const float inv = a > 1e-6f ? 1.0f / a : 0.0f;  // unpremultiply for display
    for (int c = 0; c < 3; ++c) {
      float v = b.px[i * 4 + c] * inv;
      v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
      out[i * 4 + std::size_t(c)] = (unsigned char)(v * 255.0f + 0.5f);
    }
    out[i * 4 + 3] = (unsigned char)((a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a)) * 255.0f + 0.5f);
  }
  if (!stbi_write_png(path, b.w, b.h, 4, out.data(), b.w * 4)) {
    std::fprintf(stderr, "cannot write %s\n", path);
    return false;
  }
  return true;
}

// Tint each shard a distinct colour. The same diagnostic the plugin's
// "Show Pieces" draws, and the fastest way to see what a pattern is doing.
void drawPieces(Bitmap& dst, const shp::ShatterGeometry& geom) {
  for (int y = 0; y < dst.h; ++y) {
    for (int x = 0; x < dst.w; ++x) {
      const std::size_t p = std::size_t(y) * std::size_t(dst.w) + std::size_t(x);
      const int32_t l = geom.labelImage.empty() ? 0 : geom.labelImage[p];
      float* o = &dst.px[p * 4];
      if (l == 0) {
        o[0] = o[1] = o[2] = o[3] = 0.0f;
        continue;
      }
      const uint32_t id = uint32_t(l);
      const float r = 0.25f + 0.75f * shp::Hash01(id, 11u);
      const float g = 0.25f + 0.75f * shp::Hash01(id, 22u);
      const float b = 0.25f + 0.75f * shp::Hash01(id, 33u);
      o[0] = r; o[1] = g; o[2] = b; o[3] = 1.0f;
    }
  }
}

shp::Pattern parsePattern(const std::string& s) {
  if (s == "bricks") return shp::Pattern::Bricks;
  if (s == "grid") return shp::Pattern::Grid;
  if (s == "alpha") return shp::Pattern::Alpha;
  if (s == "map") return shp::Pattern::Map;
  if (s == "manual") return shp::Pattern::Manual;
  return shp::Pattern::Voronoi;
}

// "x1,y1,x2,y2" in normalized frame coordinates.
bool parseSlice(const std::string& s, shp::Slice& out) {
  return std::sscanf(s.c_str(), "%f,%f,%f,%f", &out.x1, &out.y1, &out.x2, &out.y2) == 4;
}

shp::ForceMode parseForceMode(const std::string& s) {
  if (s == "inside") return shp::ForceMode::InsideRadius;
  if (s == "wave") return shp::ForceMode::RadialWave;
  return shp::ForceMode::AllAtStart;
}

// Frame-level averages the force and spin scaling are expressed against.
shp::FrameScale makeScale(const shp::ShatterGeometry& geom) {
  shp::FrameScale fs;
  fs.width = float(geom.width);
  fs.height = float(geom.height);

  double area = 0.0, radius = 0.0;
  for (const shp::Shard& s : geom.shards) {
    area += double(s.area);
    radius += double(s.radius);
  }
  const double n = double(geom.shards.size());
  fs.meanArea = n > 0 ? float(area / n) : 1.0f;
  fs.meanRadius = n > 0 ? float(radius / n) : 1.0f;
  return fs;
}

// Lay frames out side by side so a whole shatter is one image to look at.
void blitInto(Bitmap& sheet, const Bitmap& frame, int col, int cols, int scaleDiv) {
  const int cw = frame.w / scaleDiv;
  const int ch = frame.h / scaleDiv;
  (void)cols;
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      const std::size_t sp =
          std::size_t(y * scaleDiv) * std::size_t(frame.w) + std::size_t(x * scaleDiv);
      const std::size_t dp =
          std::size_t(y) * std::size_t(sheet.w) + std::size_t(col * cw + x);
      for (int c = 0; c < 4; ++c) sheet.px[dp * 4 + std::size_t(c)] = frame.px[sp * 4 + std::size_t(c)];
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: shattertest in.png out.png [options]\n"
                 "  --pattern voronoi|bricks|grid|alpha|map\n"
                 "  --cells N          target cell count (voronoi)\n"
                 "  --jitter F         0..1 seed scatter\n"
                 "  --rows N           brick rows\n"
                 "  --threshold F      alpha threshold\n"
                 "  --min-area F       min piece area, fraction of frame\n"
                 "  --bridge F         bridge radius, fraction of height\n"
                 "  --seed N\n"
                 "  --polys            build collision hulls and report them\n"
                 "  --pieces           draw the diagnostic tint instead of the image\n"
                 "\n"
                 "  --frames N         render N frames of shatter (default: 1 still)\n"
                 "  --strip N          contact sheet of N frames across the shatter\n"
                 "  --start N          shatter start frame\n"
                 "  --at F             render one frame at clip time F\n"
                 "  --force-mode all|inside|wave\n"
                 "  --fx F --fy F      force 1 position, normalized\n"
                 "  --radius F         force 1 radius, fraction of width\n"
                 "  --strength F\n"
                 "  --gravity F  --gravity-dir DEG\n"
                 "  --spin F  --random F  --viscosity F  --mass-var F\n");
    return 2;
  }

  const char* inPath = argv[1];
  const char* outPath = argv[2];

  shp::ShapeParams sp;
  shp::ForceParams fp;
  shp::PhysicsParams pp;
  shp::RenderParams rp;
  shp::CompositeParams cp;
  shp::BlurParams mb;

  fp.f1.enabled = true;
  int frames = 0;
  int strip = 0;
  float atTime = -1.0f;
  bool rigid = false;
  int bench = 0;
  std::vector<shp::Collider> colliders;

  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](float def) { return i + 1 < argc ? float(std::atof(argv[++i])) : def; };
    if (a == "--pattern" && i + 1 < argc) sp.pattern = parsePattern(argv[++i]);
    else if (a == "--force-mode" && i + 1 < argc) fp.mode = parseForceMode(argv[++i]);
    else if (a == "--cells") sp.cellCount = int(next(50));
    else if (a == "--jitter") sp.cellJitter = next(0.75f);
    else if (a == "--rows") sp.brickRows = int(next(8));
    else if (a == "--threshold") sp.alphaThreshold = next(0.15f);
    else if (a == "--min-area") sp.minPieceArea = next(0.0002f);
    else if (a == "--bridge") sp.bridgeRadius = next(0.0f);
    else if (a == "--seed") { sp.shapeSeed = int(next(1)); fp.seed = sp.shapeSeed; }
    else if (a == "--polys") sp.buildPolygons = true;
    else if (a == "--pieces") cp.showPieces = true;
    else if (a == "--frames") frames = int(next(24));
    else if (a == "--strip") strip = int(next(8));
    else if (a == "--at") atTime = next(0.0f);
    else if (a == "--start") fp.startFrame = int(next(0));
    else if (a == "--fx") fp.f1.posX = next(0.5f);
    else if (a == "--fy") fp.f1.posY = next(0.5f);
    else if (a == "--radius") fp.f1.radius = next(0.4f);
    else if (a == "--strength") fp.f1.strength = next(1.0f);
    else if (a == "--wave-speed") fp.waveSpeed = next(0.05f);
    else if (a == "--gravity") pp.gravity = next(3.0f);
    else if (a == "--gravity-dir") pp.gravityDirection = next(180.0f);
    else if (a == "--spin") pp.rotationSpeed = next(1.0f);
    else if (a == "--random") pp.randomness = next(0.15f);
    else if (a == "--viscosity") pp.viscosity = next(0.1f);
    else if (a == "--mass-var") pp.massVariance = next(0.3f);
    else if (a == "--rigid") { rigid = true; sp.buildPolygons = true; }
    else if (a == "--bounce") pp.bounciness = next(0.3f);
    else if (a == "--friction") pp.friction = next(0.4f);
    else if (a == "--floor") pp.floorY = next(1.0f);
    else if (a == "--no-floor") pp.floorEnabled = false;
    else if (a == "--walls") pp.wallsEnabled = true;
    else if (a == "--no-collide") pp.collideShards = false;
    else if (a == "--substeps") pp.substeps = int(next(4));
    else if (a == "--bench") bench = int(next(30));
    else if (a == "--blur") mb.enabled = true;
    else if (a == "--shutter") mb.shutterAngle = next(180.0f);
    else if (a == "--blur-samples") mb.samples = int(next(8));
    else if (a == "--dir1") { fp.f1.shape = shp::ForceShape::Directional; fp.f1.direction = next(90.0f); }
    else if (a == "--width1") fp.f1.width = next(0.5f);
    else if (a == "--delay1") fp.f1.delay = next(0.0f);
    else if (a == "--f2") fp.f2.enabled = true;
    else if (a == "--f2x") fp.f2.posX = next(0.5f);
    else if (a == "--f2y") fp.f2.posY = next(0.5f);
    else if (a == "--f2-radius") fp.f2.radius = next(0.4f);
    else if (a == "--f2-strength") fp.f2.strength = next(1.0f);
    else if (a == "--dir2") { fp.f2.shape = shp::ForceShape::Directional; fp.f2.direction = next(90.0f); }
    else if (a == "--width2") fp.f2.width = next(0.5f);
    else if (a == "--delay2") fp.f2.delay = next(0.0f);
    else if (a == "--slice" && i + 1 < argc) {
      shp::Slice sl;
      if (parseSlice(argv[++i], sl)) sp.slices.push_back(sl);
    }
    else if ((a == "--collider" || a == "--box" || a == "--circle") && i + 1 < argc) {
      shp::Collider co;
      co.shape = a == "--box"      ? shp::ColliderShape::Rect
                 : a == "--circle" ? shp::ColliderShape::Circle
                                   : shp::ColliderShape::Line;
      // The angle is optional, so "--box x1,y1,x2,y2" still means an upright one.
      const int got = std::sscanf(argv[++i], "%f,%f,%f,%f,%f", &co.x1, &co.y1, &co.x2, &co.y2,
                                  &co.angle);
      if (got >= 4) colliders.push_back(co);
    }
  }

  Bitmap src;
  if (!loadPng(inPath, src)) return 1;

  auto geom = shp::buildGeometry(src.view(), nullptr, sp);
  std::printf("%dx%d  pieces=%zu\n", geom->width, geom->height, geom->shards.size());

  if (sp.buildPolygons) {
    std::size_t withPoly = 0;
    double totalVerts = 0;
    for (const shp::Shard& s : geom->shards) {
      if (s.poly.size() >= 3) {
        ++withPoly;
        totalVerts += double(s.poly.size());
      }
    }
    std::printf("hulls=%zu  avg verts=%.1f\n", withPoly,
                withPoly ? totalVerts / double(withPoly) : 0.0);
  }

  // A still tint, the old behaviour, when no timing was asked for.
  if (frames == 0 && strip == 0 && bench == 0 && atTime < 0.0f) {
    Bitmap out;
    out.w = src.w;
    out.h = src.h;
    out.px.assign(std::size_t(out.w) * std::size_t(out.h) * 4, 0.0f);
    drawPieces(out, *geom);
    if (!savePng(outPath, out)) return 1;
    std::printf("wrote %s\n", outPath);
    return 0;
  }

  const shp::FrameScale fs = makeScale(*geom);
  std::vector<shp::ShardPose> poses;
  std::vector<shp::ShardDraw> draws;
  shp::TileGrid tiles;
  shp::SimCache sim;

  auto renderAt = [&](float t, Bitmap& out) {
    out.w = src.w;
    out.h = src.h;
    out.px.assign(std::size_t(out.w) * std::size_t(out.h) * 4, 0.0f);
    if (rigid) {
      sim.getPoses(geom, fp, pp, rp, fs, colliders, t, poses);
    } else {
      shp::EvaluateBallistic(geom->shards, fp, pp, rp, fs, t, poses);
    }
    shp::BuildDrawList(*geom, poses, mb, draws);
    shp::BuildTileGrid(draws, geom->width, geom->height, tiles);
    shp::Composite(out.mutableView(), src.view(), *geom, draws, tiles, cp);
  };

  // Timing without any PNG encoding in the way. A 4K stb_write_png costs
  // several hundred milliseconds, which is enough to hide the thing being
  // measured entirely.
  if (bench > 0) {
    Bitmap frame;
    renderAt(0.0f, frame);  // warm the caches and the geometry

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench; ++i) renderAt(float(fp.startFrame + i), frame);
    const auto t1 = std::chrono::steady_clock::now();

    const double ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    std::printf("%d frames of %dx%d in %.0f ms -> %.1f ms/frame (%.1f fps)\n", bench, src.w,
                src.h, ms, ms / double(bench), 1000.0 * double(bench) / ms);
    return 0;
  }

  if (strip > 0) {
    const int total = frames > 0 ? frames : 24;
    const int scaleDiv = 2;
    Bitmap sheet;
    sheet.w = (src.w / scaleDiv) * strip;
    sheet.h = src.h / scaleDiv;
    sheet.px.assign(std::size_t(sheet.w) * std::size_t(sheet.h) * 4, 0.0f);

    Bitmap frame;
    for (int i = 0; i < strip; ++i) {
      const float t = float(fp.startFrame) + float(total) * float(i) / float(strip - 1 > 0 ? strip - 1 : 1);
      renderAt(t, frame);
      blitInto(sheet, frame, i, strip, scaleDiv);
    }
    if (!savePng(outPath, sheet)) return 1;
    std::printf("wrote %s (%d frames across %d)\n", outPath, strip, total);
    return 0;
  }

  if (frames > 0) {
    std::string base = outPath;
    const std::size_t dot = base.rfind('.');
    const std::string stem = dot == std::string::npos ? base : base.substr(0, dot);
    const std::string ext = dot == std::string::npos ? ".png" : base.substr(dot);

    Bitmap frame;
    char path[1024];
    for (int i = 0; i < frames; ++i) {
      renderAt(float(fp.startFrame + i), frame);
      std::snprintf(path, sizeof(path), "%s_%04d%s", stem.c_str(), i, ext.c_str());
      if (!savePng(path, frame)) return 1;
    }
    std::printf("wrote %d frames as %s_NNNN%s\n", frames, stem.c_str(), ext.c_str());
    return 0;
  }

  Bitmap out;
  renderAt(atTime, out);
  if (!savePng(outPath, out)) return 1;
  std::printf("wrote %s at t=%.1f\n", outPath, atTime);
  return 0;
}
