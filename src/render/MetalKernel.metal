// The Metal compositor.
//
// A line-for-line port of ShadePixel and the helpers it calls, from
// core/Compositor.h, core/Sampler.h, core/TransformMath.h and core/Hash.h.
// Unlike CudaKernel.cu this file DOES contain maths of its own, because Metal
// Shading Language cannot include those headers -- see MetalRender.h. Every
// function below names its twin, and keeps the twin's operation order so the
// two round the same way; tests/metalparity.cpp checks that they still agree.
//
// Not compiled at build time. It is embedded into the plugin as a string and
// compiled once per GPU when the plugin first needs it, which keeps the build
// free of a dependency on full Xcode (the offline `metal` compiler does not ship
// with the Command Line Tools). Fast maths is switched off at that compile, for
// the same reason the CPU build avoids -ffast-math.
//
// SHP_MAX_BLUR_SAMPLES is injected by the host from kMaxBlurSamples, so the
// ShardDraw layout below cannot drift from the C++ one; the host also
// static_asserts the C++ layout this struct assumes.

#include <metal_stdlib>
using namespace metal;

#ifndef SHP_MAX_BLUR_SAMPLES
#error "SHP_MAX_BLUR_SAMPLES must be defined by the host"
#endif

constant float kPi = 3.14159265358979323846f;

// ------------------------------------------------------------------ layouts
// Every member is a 4-byte scalar, so MSL and C++ lay these out identically.

struct Mat3 {                 // TransformMath.h: Mat3
  float m[6];
};

struct ShardDraw {            // Compositor.h: ShardDraw
  Mat3 inv[SHP_MAX_BLUR_SAMPLES];
  int count;
  int label;
  float alpha;
  int x1, y1, x2, y2;
  float z;
  float shade;
  float angle;
};

struct ShadeArgs {            // MetalRender.mm: ShadeArgs
  int labelW, labelH;
  int srcW, srcH;
  int srcTop, srcStride;      // floats: offset of the TOP row, and a signed stride
  int dstTop, dstStride;
  int dstW, dstH;
  int tilesX, tilesY, tileSize;
  int edgeAA, showPieces;
  float bevel, bevelWidth, lightX, lightY;
};

struct HashArgs {             // MetalRender.mm: HashArgs
  int width, height, stride;  // stride in floats, positive, bottom-up
  float threshold;
};

// ------------------------------------------------------------------ Hash.h

inline uint Hash32(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

inline uint Hash32(uint a, uint b) {
  return Hash32(a ^ (Hash32(b) + 0x9e3779b9u + (a << 6) + (a >> 2)));
}

inline float Hash01(uint a, uint b) {
  return float(Hash32(a, b) >> 8) * (1.0f / 16777216.0f);
}

// ------------------------------------------------------------ TransformMath.h

inline float Lerp(float a, float b, float t) { return a + (b - a) * t; }

inline float Clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Mat3::Apply
inline void Apply(device const Mat3& M, float x, float y, thread float& outX,
                  thread float& outY) {
  outX = M.m[0] * x + M.m[1] * y + M.m[2];
  outY = M.m[3] * x + M.m[4] * y + M.m[5];
}

// ----------------------------------------------------------------- Sampler.h
// Only the kEdgeBlack mode exists here, because it is the only one ShadePixel
// uses: outside the source is transparent black.

inline float4 FetchTexel(device const float* src, constant ShadeArgs& a, int x, int y) {
  if (x < 0 || x >= a.srcW || y < 0 || y >= a.srcH) return float4(0.0f);
  const long p = long(a.srcTop) + long(y) * long(a.srcStride) + long(x) * 4;
  return float4(src[p + 0], src[p + 1], src[p + 2], src[p + 3]);
}

inline float4 SampleBilinear(device const float* src, constant ShadeArgs& a, float sx,
                             float sy) {
  const float fx = sx - 0.5f;
  const float fy = sy - 0.5f;

  const int x0 = int(floor(fx));
  const int y0 = int(floor(fy));
  const float tx = fx - float(x0);
  const float ty = fy - float(y0);

  const float4 p00 = FetchTexel(src, a, x0, y0);
  const float4 p10 = FetchTexel(src, a, x0 + 1, y0);
  const float4 p01 = FetchTexel(src, a, x0, y0 + 1);
  const float4 p11 = FetchTexel(src, a, x0 + 1, y0 + 1);

  float4 out;
  for (int c = 0; c < 4; ++c) {
    const float top = Lerp(p00[c], p10[c], tx);
    const float bottom = Lerp(p01[c], p11[c], tx);
    out[c] = Lerp(top, bottom, ty);
  }
  return out;
}

// -------------------------------------------------------------- Compositor.h

// StencilHit
inline bool StencilHit(device const int* labels, int lw, int lh, float sx, float sy,
                       int label) {
  if (sx < 0.0f || sy < 0.0f) return false;
  const int ix = int(sx);
  const int iy = int(sy);
  if (ix >= lw || iy >= lh) return false;
  return labels[long(iy) * lw + ix] == label;
}

// BevelShade. The caller only invokes it with a real edge field.
inline float BevelShade(device const uchar* edgeDist, device const uchar* edgeAngle, int lw,
                        int ix, int iy, float shardAngle, float strength, float widthPx,
                        float lightX, float lightY) {
  if (strength <= 0.0f || widthPx <= 0.01f) return 1.0f;

  const long p = long(iy) * lw + ix;
  const float d = float(edgeDist[p]);
  if (d >= widthPx) return 1.0f;

  const float t = 1.0f - d / widthPx;
  const float falloff = t * t;

  const float ang = float(edgeAngle[p]) * (2.0f * kPi / 255.0f) + shardAngle;
  const float nx = precise::cos(ang);
  const float ny = precise::sin(ang);

  const float lambert = nx * lightX + ny * lightY;
  const float amount = lambert > 0.0f ? lambert * 0.6f : lambert;
  return Clampf(1.0f + strength * falloff * amount, 0.0f, 4.0f);
}

// ShadePixel, plus the tile lookup Composite() does around it.
kernel void shp_shade(device const int* labels [[buffer(0)]],
                      device const uchar* edgeDist [[buffer(1)]],
                      device const uchar* edgeAngle [[buffer(2)]],
                      device const float* src [[buffer(3)]],
                      device float* dst [[buffer(4)]],
                      device const ShardDraw* draws [[buffer(5)]],
                      device const int* tileStart [[buffer(6)]],
                      device const int* tileShardIdx [[buffer(7)]],
                      constant ShadeArgs& a [[buffer(8)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const int x = int(gid.x);
  const int y = int(gid.y);
  if (x >= a.dstW || y >= a.dstH) return;

  float4 out = float4(0.0f);

  const int tx = x / a.tileSize;
  const int ty = y / a.tileSize;
  if (tx < a.tilesX && ty < a.tilesY) {
    const int tile = ty * a.tilesX + tx;
    const int lo = tileStart[tile];
    const int hi = tileStart[tile + 1];

    const float dx = float(x) + 0.5f;
    const float dy = float(y) + 0.5f;
    const int lw = a.labelW, lh = a.labelH;

    for (int k = lo; k < hi; ++k) {
      device const ShardDraw& d = draws[tileShardIdx[k]];
      if (x < d.x1 || x >= d.x2 || y < d.y1 || y >= d.y2) continue;

      // Shutter samples accumulate and are averaged; see ShadePixel for why
      // they must not be composited over one another.
      float4 acc = float4(0.0f);
      bool any = false;

      for (int s = 0; s < d.count; ++s) {
        float sx, sy;
        Apply(d.inv[s], dx, dy, sx, sy);

        float cov = 0.0f;
        if (a.edgeAA != 0) {
          const float off[4][2] = {
              {-0.25f, -0.25f}, {0.25f, -0.25f}, {-0.25f, 0.25f}, {0.25f, 0.25f}};
          for (int q = 0; q < 4; ++q) {
            float ax, ay;
            Apply(d.inv[s], dx + off[q][0], dy + off[q][1], ax, ay);
            if (StencilHit(labels, lw, lh, ax, ay, d.label)) cov += 0.25f;
          }
        } else {
          cov = StencilHit(labels, lw, lh, sx, sy, d.label) ? 1.0f : 0.0f;
        }
        if (cov <= 0.0f) continue;

        float4 px;
        if (a.showPieces != 0) {
          const uint id = uint(d.label);
          px = float4(0.25f + 0.75f * Hash01(id, 11u), 0.25f + 0.75f * Hash01(id, 22u),
                      0.25f + 0.75f * Hash01(id, 33u), 1.0f);
        } else {
          px = SampleBilinear(src, a, sx, sy);
        }

        // Colour only, never alpha -- see ShadePixel.
        float lum = d.shade;
        if (a.bevel > 0.0f) {
          const int bx = int(sx), by = int(sy);
          if (sx >= 0.0f && sy >= 0.0f && bx < lw && by < lh)
            lum *= BevelShade(edgeDist, edgeAngle, lw, bx, by, d.angle, a.bevel,
                              a.bevelWidth, a.lightX, a.lightY);
        }

        const float al = cov * d.alpha;
        acc[0] += px[0] * al * lum;
        acc[1] += px[1] * al * lum;
        acc[2] += px[2] * al * lum;
        acc[3] += px[3] * al;
        any = true;
      }

      if (!any) continue;

      const float inv = 1.0f / float(d.count);
      acc[0] *= inv;
      acc[1] *= inv;
      acc[2] *= inv;
      acc[3] *= inv;

      // CompositeOver (Sampler.h): premultiplied src over dst.
      const float ia = 1.0f - acc[3];
      out[0] = acc[0] + out[0] * ia;
      out[1] = acc[1] + out[1] * ia;
      out[2] = acc[2] + out[2] * ia;
      out[3] = acc[3] + out[3] * ia;
    }
  }

  const long o = long(a.dstTop) + long(y) * long(a.dstStride) + long(x) * 4;
  dst[o + 0] = out[0];
  dst[o + 1] = out[1];
  dst[o + 2] = out[2];
  dst[o + 3] = out[3];
}

// Order-independent content fingerprint: CudaKernel.cu's kHash, bit for bit.
// Its 64-bit atomicXor becomes two 32-bit halves, since Metal has no 64-bit
// atomic XOR; XOR is per bit, so the result is the same.
kernel void shp_hash(device const float* src [[buffer(0)]],
                     device atomic_uint* out [[buffer(1)]],
                     constant HashArgs& a [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
  const int x = int(gid.x) * 8;
  const int y = int(gid.y) * 8;
  if (x >= a.width || y >= a.height) return;

  const float al = src[long(y) * long(a.stride) + long(x) * 4 + 3];
  const uint q = uint(fmin(15.0f, fmax(0.0f, al) * 15.0f + 0.5f));
  const uint bucket = q + (al > a.threshold ? 16u : 0u);

  const uint lo = Hash32((uint(x) * 73856093u) ^ (uint(y) * 19349663u) ^ bucket);
  const uint hi = Hash32(bucket, uint(x) + uint(y) * 7919u);
  atomic_fetch_xor_explicit(&out[0], lo, memory_order_relaxed);
  atomic_fetch_xor_explicit(&out[1], hi, memory_order_relaxed);
}
