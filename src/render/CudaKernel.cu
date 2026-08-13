// The CUDA compositor.
//
// This file contains NO maths of its own. It includes core/Compositor.h and
// calls the same ShadePixel the CPU path calls, compiled here as device code via
// the SHP_HD macro. Everything below is memory management, launch configuration
// and transfers -- if a change to how a pixel is shaded ever needs to be made
// here, it belongs in the header instead.

#include <cuda_runtime.h>

#include <cstdio>

#include "CudaRender.h"

namespace shp {
namespace {

bool ok(cudaError_t e) { return e == cudaSuccess; }

/// A device allocation that grows and is never shrunk.
///
/// cudaMalloc synchronises the device, which at tens of microseconds a call is
/// enough to show up when it happens per frame. Keeping the buffers resident and
/// only growing them means the steady state allocates nothing at all.
struct DeviceBuffer {
  void* p = nullptr;
  std::size_t bytes = 0;

  bool ensure(std::size_t need) {
    if (need <= bytes) return true;
    if (p) cudaFree(p);
    p = nullptr;
    bytes = 0;
    if (!ok(cudaMalloc(&p, need))) return false;
    bytes = need;
    return true;
  }

  void release() {
    if (p) cudaFree(p);
    p = nullptr;
    bytes = 0;
  }
};

}  // namespace

struct CudaContext {
  DeviceBuffer labels;   ///< int32 per pixel, resident across frames
  int labelW = 0, labelH = 0;

  DeviceBuffer draws;        ///< ShardDraw array, uploaded per frame
  DeviceBuffer tileStart;    ///< CSR offsets
  DeviceBuffer tileShardIdx; ///< CSR payload
  DeviceBuffer hash;         ///< 8 bytes of accumulator
  DeviceBuffer staging;      ///< only used by the host-sync parity entry

  ~CudaContext() {
    labels.release();
    draws.release();
    tileStart.release();
    tileShardIdx.release();
    hash.release();
    staging.release();
  }
};

namespace {

// ------------------------------------------------------------------ kernels

__global__ void kShade(const int32_t* labels, int lw, int lh, SampleView src, float* dst,
                       long long dstStride, int w, int h, const ShardDraw* draws,
                       const int32_t* tileStart, const int32_t* tileShardIdx, int tilesX,
                       int tilesY, int edgeAA, int showPieces) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  float* out = dst + (long long)(y) * dstStride + (long long)(x) * 4;

  const int tx = x / kTileSize;
  const int ty = y / kTileSize;
  if (tx >= tilesX || ty >= tilesY) {
    out[0] = out[1] = out[2] = out[3] = 0.0f;
    return;
  }

  const int tile = ty * tilesX + tx;
  const int32_t lo = tileStart[tile];
  const int32_t hi = tileStart[tile + 1];
  if (lo == hi) {
    out[0] = out[1] = out[2] = out[3] = 0.0f;
    return;
  }

  // The identical function the CPU path calls.
  ShadePixel(labels, lw, lh, src, draws, tileShardIdx, lo, hi, x, y, edgeAA != 0,
             showPieces != 0, out);
}

/// Order-independent content fingerprint.
///
/// Every sampled pixel is folded into an avalanche hash and the results are
/// XOR-accumulated, so the answer does not depend on which thread finished
/// first. That is deliberately a different function from the host's FNV-1a
/// chain; see the note in CudaRender.h about why the two are never compared.
__global__ void kHash(const float* src, int w, int h, long long stride, float threshold,
                      unsigned long long* out) {
  const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 8;
  if (x >= w || y >= h) return;

  const float a = src[(long long)(y) * stride + (long long)(x) * 4 + 3];
  // Rounded, not truncated, for the same reason as the host hash: truncation
  // makes bucket 15 reachable only by exactly 1.0, so the commonest value in an
  // alpha channel sits alone on a boundary.
  const unsigned q = (unsigned)fminf(15.0f, fmaxf(0.0f, a) * 15.0f + 0.5f);
  const unsigned bucket = q + (a > threshold ? 16u : 0u);

  unsigned long long v = (unsigned long long)Hash32((uint32_t)(x * 73856093) ^
                                                    (uint32_t)(y * 19349663) ^ bucket);
  v |= (unsigned long long)Hash32(bucket, (uint32_t)(x + y * 7919)) << 32;
  atomicXor(out, v);
}

}  // namespace

// ---------------------------------------------------------------------------

bool CudaAvailable() {
  static int cached = -1;
  if (cached >= 0) return cached != 0;
  int count = 0;
  cached = (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) ? 1 : 0;
  return cached != 0;
}

CudaContext* CudaCreate() { return new CudaContext(); }
void CudaDestroy(CudaContext* ctx) { delete ctx; }

std::uint64_t CudaHashSource(CudaContext* ctx, const float* devSrc, int width, int height,
                             std::ptrdiff_t rowStrideFloats, float threshold, void* stream) {
  if (!ctx || !devSrc || width <= 0 || height <= 0) return 0;
  if (!ctx->hash.ensure(sizeof(unsigned long long))) return 0;

  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (!ok(cudaMemsetAsync(ctx->hash.p, 0, sizeof(unsigned long long), s))) return 0;

  const int sw = (width + 7) / 8;
  const int sh = (height + 7) / 8;
  const dim3 block(16, 16);
  const dim3 grid((sw + block.x - 1) / block.x, (sh + block.y - 1) / block.y);
  kHash<<<grid, block, 0, s>>>(devSrc, width, height, (long long)rowStrideFloats, threshold,
                               static_cast<unsigned long long*>(ctx->hash.p));

  unsigned long long out = 0;
  if (!ok(cudaMemcpyAsync(&out, ctx->hash.p, sizeof(out), cudaMemcpyDeviceToHost, s))) return 0;
  cudaStreamSynchronize(s);  // eight bytes; this is the whole point of hashing on device

  // Fold in the dimensions, so a resize invalidates even if the content sum
  // happened to land the same way.
  out ^= (unsigned long long)(width) * 0x9e3779b97f4a7c15ull;
  out ^= (unsigned long long)(height) * 0xc2b2ae3d27d4eb4full;
  // Never return 0: the caller treats it as "no answer".
  return out ? out : 1ull;
}

bool CudaDownload(CudaContext* ctx, const float* devSrc, int width, int height,
                  std::ptrdiff_t rowStrideFloats, float* hostDst, void* stream) {
  if (!ctx || !devSrc || !hostDst) return false;
  cudaStream_t s = static_cast<cudaStream_t>(stream);

  const std::size_t rowBytes = std::size_t(width) * 4 * sizeof(float);
  if (!ok(cudaMemcpy2DAsync(hostDst, rowBytes, devSrc,
                            std::size_t(rowStrideFloats) * sizeof(float), rowBytes,
                            std::size_t(height), cudaMemcpyDeviceToHost, s)))
    return false;
  return ok(cudaStreamSynchronize(s));
}

bool CudaUploadLabels(CudaContext* ctx, const std::int32_t* labels, int width, int height,
                      void* stream) {
  if (!ctx || !labels) return false;
  const std::size_t n = std::size_t(width) * std::size_t(height) * sizeof(std::int32_t);
  if (!ctx->labels.ensure(n)) return false;

  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (!ok(cudaMemcpyAsync(ctx->labels.p, labels, n, cudaMemcpyHostToDevice, s))) return false;
  ctx->labelW = width;
  ctx->labelH = height;
  return true;
}

bool CudaComposite(CudaContext* ctx, const float* devSrc, float* devDst, int width, int height,
                   std::ptrdiff_t srcStrideFloats, std::ptrdiff_t dstStrideFloats,
                   const ShardDraw* draws, int drawCount, const std::int32_t* tileStart,
                   int tileCount, const std::int32_t* tileShardIdx, int tileShardCount,
                   int tilesX, int tilesY, const CompositeParams& params, void* stream) {
  if (!ctx || !devSrc || !devDst) return false;
  if (ctx->labels.p == nullptr) return false;

  cudaStream_t s = static_cast<cudaStream_t>(stream);

  // The CSR arrays and the draw list are small -- a few hundred kilobytes for a
  // thousand shards -- so they are uploaded whole every frame rather than
  // diffed. The frame itself never moves, which is the point.
  const std::size_t drawBytes = std::size_t(drawCount > 0 ? drawCount : 1) * sizeof(ShardDraw);
  const std::size_t startBytes = std::size_t(tileCount + 1) * sizeof(std::int32_t);
  const std::size_t idxBytes =
      std::size_t(tileShardCount > 0 ? tileShardCount : 1) * sizeof(std::int32_t);

  if (!ctx->draws.ensure(drawBytes)) return false;
  if (!ctx->tileStart.ensure(startBytes)) return false;
  if (!ctx->tileShardIdx.ensure(idxBytes)) return false;

  if (drawCount > 0 &&
      !ok(cudaMemcpyAsync(ctx->draws.p, draws, std::size_t(drawCount) * sizeof(ShardDraw),
                          cudaMemcpyHostToDevice, s)))
    return false;
  if (!ok(cudaMemcpyAsync(ctx->tileStart.p, tileStart, startBytes, cudaMemcpyHostToDevice, s)))
    return false;
  if (tileShardCount > 0 &&
      !ok(cudaMemcpyAsync(ctx->tileShardIdx.p, tileShardIdx,
                          std::size_t(tileShardCount) * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice, s)))
    return false;

  SampleView sv;
  sv.data = devSrc;
  sv.width = width;
  sv.height = height;
  sv.rowStrideFloats = int(srcStrideFloats);

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  kShade<<<grid, block, 0, s>>>(
      static_cast<const int32_t*>(ctx->labels.p), ctx->labelW, ctx->labelH, sv, devDst,
      (long long)dstStrideFloats, width, height, static_cast<const ShardDraw*>(ctx->draws.p),
      static_cast<const int32_t*>(ctx->tileStart.p),
      static_cast<const int32_t*>(ctx->tileShardIdx.p), tilesX, tilesY,
      params.edgeAntialias ? 1 : 0, params.showPieces ? 1 : 0);

  return ok(cudaGetLastError());
}

bool CudaCompositeHostSync(const float* hostSrc, float* hostDst, const std::int32_t* labels,
                           int width, int height, const ShardDraw* draws, int drawCount,
                           const std::int32_t* tileStart, int tileCount,
                           const std::int32_t* tileShardIdx, int tileShardCount, int tilesX,
                           int tilesY, const CompositeParams& params) {
  if (!CudaAvailable()) return false;

  CudaContext ctx;
  const std::size_t frameFloats = std::size_t(width) * std::size_t(height) * 4;
  const std::size_t frameBytes = frameFloats * sizeof(float);

  DeviceBuffer devSrc, devDst;
  if (!devSrc.ensure(frameBytes) || !devDst.ensure(frameBytes)) return false;
  if (!ok(cudaMemcpy(devSrc.p, hostSrc, frameBytes, cudaMemcpyHostToDevice))) return false;
  if (!ok(cudaMemset(devDst.p, 0, frameBytes))) return false;

  if (!CudaUploadLabels(&ctx, labels, width, height, nullptr)) return false;
  cudaStreamSynchronize(nullptr);

  // Host buffers here are packed top-down, so the strides are positive. In the
  // plugin they are negative, pointing at the top row of a bottom-up buffer --
  // the arithmetic is identical either way, which is exactly what makes this a
  // fair parity comparison.
  const std::ptrdiff_t stride = std::ptrdiff_t(width) * 4;
  if (!CudaComposite(&ctx, static_cast<const float*>(devSrc.p), static_cast<float*>(devDst.p),
                     width, height, stride, stride, draws, drawCount, tileStart, tileCount,
                     tileShardIdx, tileShardCount, tilesX, tilesY, params, nullptr))
    return false;

  if (!ok(cudaStreamSynchronize(nullptr))) return false;
  const bool copied = ok(cudaMemcpy(hostDst, devDst.p, frameBytes, cudaMemcpyDeviceToHost));

  devSrc.release();
  devDst.release();
  return copied;
}

}  // namespace shp
