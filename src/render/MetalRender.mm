// The Metal host side: pipeline compilation, buffers and dispatch.
//
// Compiled with ARC. Every id<> held in a C++ struct below is a strong
// reference, which is what keeps a buffer alive while a command buffer that
// uses it is still in flight -- Metal retains bound resources until the command
// buffer completes, so replacing ctx->labels mid-flight is safe.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <os/log.h>

#include <cstddef>
#include <cstring>
#include <mutex>
#include <vector>

#include "MetalKernelSource.h"  // generated: kMetalKernelSource
#include "MetalRender.h"

namespace shp {

// The kernel mirrors ShardDraw member for member; these pin the layout it
// assumes. If one fires, MetalKernel.metal's struct needs the same change.
static_assert(sizeof(Mat3) == 6 * sizeof(float), "Mat3 layout changed");
static_assert(offsetof(ShardDraw, inv) == 0, "ShardDraw layout changed");
static_assert(offsetof(ShardDraw, count) == sizeof(Mat3) * kMaxBlurSamples,
              "ShardDraw layout changed");
static_assert(offsetof(ShardDraw, label) == offsetof(ShardDraw, count) + 4,
              "ShardDraw layout changed");
static_assert(offsetof(ShardDraw, alpha) == offsetof(ShardDraw, count) + 8,
              "ShardDraw layout changed");
static_assert(offsetof(ShardDraw, x1) == offsetof(ShardDraw, count) + 12,
              "ShardDraw layout changed");
static_assert(offsetof(ShardDraw, z) == offsetof(ShardDraw, count) + 28,
              "ShardDraw layout changed");
static_assert(offsetof(ShardDraw, angle) == offsetof(ShardDraw, count) + 36,
              "ShardDraw layout changed");
static_assert(sizeof(ShardDraw) == offsetof(ShardDraw, count) + 40, "ShardDraw layout changed");

namespace {

// Mirrored in MetalKernel.metal. All 4-byte scalars, so the layouts agree.
struct ShadeArgs {
  int labelW, labelH;
  int srcW, srcH;
  int srcTop, srcStride;
  int dstTop, dstStride;
  int dstW, dstH;
  int tilesX, tilesY, tileSize;
  int edgeAA, showPieces;
  float bevel, bevelWidth, lightX, lightY;
};

struct HashArgs {
  int width, height, stride;
  float threshold;
};

// The unified system log, not stderr: Resolve discards a plugin's stderr, and a
// black frame with no explanation anywhere is the worst way for this to fail.
//   log show --last 10m --predicate 'subsystem == "com.johnlwi.ShatterPhysics"'
os_log_t metalLog() {
  static os_log_t log = os_log_create("com.johnlwi.ShatterPhysics", "metal");
  return log;
}

void logError(const char* what, NSError* err) {
  os_log_error(metalLog(), "%{public}s%{public}s%{public}s", what, err ? ": " : "",
               err ? err.localizedDescription.UTF8String : "");
}

// ---------------------------------------------------------------- pipelines

struct Pipelines {
  id<MTLDevice> device;
  id<MTLComputePipelineState> shade;  // nil when compilation failed
  id<MTLComputePipelineState> hash;
};

/// Compiled once per GPU and kept for the life of the process.
///
/// Per device rather than global because the queue Resolve hands over belongs
/// to whichever GPU it renders on, which on a Mac with more than one is not
/// necessarily the system default.
const Pipelines* pipelinesFor(id<MTLDevice> device) {
  if (!device) return nullptr;

  static std::mutex mutex;
  // Leaked on purpose: destroying ObjC objects from a static destructor while
  // the host is unloading plugins is asking for trouble, and a process-lifetime
  // cache needs no teardown.
  static auto* cache = new std::vector<Pipelines>();

  std::lock_guard<std::mutex> lock(mutex);
  for (const Pipelines& p : *cache)
    if (p.device == device) return p.shade ? &p : nullptr;

  Pipelines p;
  p.device = device;

  @autoreleasepool {
    MTLCompileOptions* opts = [MTLCompileOptions new];
    // Precise maths, for the same reason the CPU build has no -ffast-math: the
    // NaN guards upstream must survive, and the parity test needs the GPU to
    // round the way the CPU does.
    if (@available(macOS 15.0, *)) {
      opts.mathMode = MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    opts.preprocessorMacros = @{@"SHP_MAX_BLUR_SAMPLES" : @(kMaxBlurSamples)};

    NSError* err = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalKernelSource]
                             options:opts
                               error:&err];
    if (!lib) {
      logError("kernel failed to compile", err);
    } else {
      id<MTLFunction> fShade = [lib newFunctionWithName:@"shp_shade"];
      id<MTLFunction> fHash = [lib newFunctionWithName:@"shp_hash"];
      if (fShade && fHash) {
        p.shade = [device newComputePipelineStateWithFunction:fShade error:&err];
        if (p.shade) p.hash = [device newComputePipelineStateWithFunction:fHash error:&err];
        if (!p.shade || !p.hash) {
          logError("pipeline creation failed", err);
          p.shade = nil;
          p.hash = nil;
        }
      } else {
        logError("kernel entry points missing", nil);
      }
    }
  }

  cache->push_back(p);
  return cache->back().shade ? &cache->back() : nullptr;
}

/// Storage for buffers the CPU fills once and the GPU reads. Shared is free on
/// Apple Silicon, where memory is unified; on an Intel Mac with a discrete GPU
/// Managed keeps a copy in VRAM instead of reading system memory over the bus.
MTLResourceOptions uploadOptions(id<MTLDevice> device) {
  return device.hasUnifiedMemory ? MTLResourceStorageModeShared
                                 : MTLResourceStorageModeManaged;
}

/// Metal refuses zero-length buffers, so an empty array gets a one-element one.
id<MTLBuffer> uploadBuffer(id<MTLDevice> device, const void* data, std::size_t bytes) {
  static const std::int32_t zero = 0;
  if (!data || bytes == 0) return [device newBufferWithBytes:&zero
                                                      length:sizeof(zero)
                                                     options:uploadOptions(device)];
  return [device newBufferWithBytes:data length:bytes options:uploadOptions(device)];
}

/// The bytes an image actually spans in its buffer, or 0 if the description
/// cannot be right. Checked against the buffer's real length before every
/// dispatch: if a host ever lays its buffers out differently from what this
/// assumes, the result is a black frame and a log line, not a GPU reading past
/// the end of an allocation.
std::size_t spanBytes(const MetalImage& img) {
  if (!img.buffer || img.width <= 0 || img.height <= 0) return 0;
  if (img.rowStrideFloats < std::ptrdiff_t(img.width) * 4) return 0;
  const std::size_t rowBytes = std::size_t(img.rowStrideFloats) * sizeof(float);
  return std::size_t(img.height - 1) * rowBytes + std::size_t(img.width) * 4 * sizeof(float);
}

bool fits(const MetalImage& img, const char* which) {
  const std::size_t need = spanBytes(img);
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)img.buffer;
  // The kernel addresses pixels with 32-bit float offsets.
  const bool ok = need > 0 && buf && need <= buf.length && need / sizeof(float) < 0x7fffffffu;
  if (!ok) {
    os_log_error(metalLog(),
                 "%{public}s buffer does not match its bounds "
                 "(%dx%d, stride %td floats, %zu bytes needed, %zu available)",
                 which, img.width, img.height, img.rowStrideFloats, need,
                 buf ? std::size_t(buf.length) : std::size_t(0));
  }
  return ok;
}

MTLSize groupsFor(int w, int h, MTLSize group) {
  return MTLSizeMake((NSUInteger(w) + group.width - 1) / group.width,
                     (NSUInteger(h) + group.height - 1) / group.height, 1);
}

bool waitOk(id<MTLCommandBuffer> cb, const char* what) {
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted) {
    logError(what, cb.error);
    return false;
  }
  return true;
}

}  // namespace

struct MetalContext {
  std::mutex mutex;  ///< one render per instance at a time, but be certain
  id<MTLDevice> device;  ///< the GPU the resident buffers live on
  id<MTLBuffer> labels;
  id<MTLBuffer> edgeDist;
  id<MTLBuffer> edgeAngle;
  int labelW = 0, labelH = 0;
};

// ---------------------------------------------------------------------------

bool MetalAvailable() {
  static const bool available = [] {
    @autoreleasepool {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      return pipelinesFor(device) != nullptr;
    }
  }();
  return available;
}

MetalContext* MetalCreate() { return new MetalContext(); }
void MetalDestroy(MetalContext* ctx) { delete ctx; }

std::uint64_t MetalHashSource(MetalContext* ctx, void* cmdQueue, const MetalImage& src,
                              float threshold) {
  if (!ctx || !cmdQueue || !fits(src, "source")) return 0;
  @autoreleasepool {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)cmdQueue;
    const Pipelines* p = pipelinesFor(queue.device);
    if (!p) return 0;

    // A fresh eight bytes per call rather than one kept in the context, so no
    // two renders can ever be clearing and reading the same accumulator.
    id<MTLBuffer> acc = [queue.device newBufferWithLength:2 * sizeof(std::uint32_t)
                                                  options:MTLResourceStorageModeShared];
    if (!acc) return 0;
    std::memset(acc.contents, 0, 2 * sizeof(std::uint32_t));

    HashArgs args{src.width, src.height, int(src.rowStrideFloats), threshold};

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p->hash];
    [enc setBuffer:(__bridge id<MTLBuffer>)src.buffer offset:0 atIndex:0];
    [enc setBuffer:acc offset:0 atIndex:1];
    [enc setBytes:&args length:sizeof(args) atIndex:2];
    const MTLSize group = MTLSizeMake(16, 16, 1);
    [enc dispatchThreadgroups:groupsFor((src.width + 7) / 8, (src.height + 7) / 8, group)
        threadsPerThreadgroup:group];
    [enc endEncoding];
    // Eight bytes; waiting for them is the whole point of hashing on the GPU.
    if (!waitOk(cb, "hash failed")) return 0;

    const auto* halves = static_cast<const std::uint32_t*>(acc.contents);
    std::uint64_t out = std::uint64_t(halves[0]) | (std::uint64_t(halves[1]) << 32);
    // Same folding as CudaHashSource: a resize invalidates even if the content
    // sum happened to land the same way.
    out ^= std::uint64_t(src.width) * 0x9e3779b97f4a7c15ull;
    out ^= std::uint64_t(src.height) * 0xc2b2ae3d27d4eb4full;
    return out ? out : 1ull;  // 0 means "no answer"
  }
}

bool MetalDownload(MetalContext* ctx, void* cmdQueue, const MetalImage& src, float* hostDst) {
  if (!ctx || !cmdQueue || !hostDst || !fits(src, "source")) return false;
  @autoreleasepool {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)cmdQueue;
    const std::size_t span = spanBytes(src);

    // Through a blit into a buffer of our own, rather than reading the host's
    // buffer directly: it may be Private, which the CPU cannot see at all, and
    // even a Shared one may still have Resolve's own GPU work pending on it.
    // Queuing the copy behind that work is what makes the read safe.
    id<MTLBuffer> staging = [queue.device newBufferWithLength:span
                                                      options:MTLResourceStorageModeShared];
    if (!staging) return false;

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)src.buffer
            sourceOffset:0
                toBuffer:staging
       destinationOffset:0
                    size:span];
    [blit endEncoding];
    if (!waitOk(cb, "download failed")) return false;

    const auto* base = static_cast<const char*>(staging.contents);
    const std::size_t rowBytes = std::size_t(src.width) * 4 * sizeof(float);
    const std::size_t strideBytes = std::size_t(src.rowStrideFloats) * sizeof(float);
    for (int y = 0; y < src.height; ++y)
      std::memcpy(reinterpret_cast<char*>(hostDst) + std::size_t(y) * rowBytes,
                  base + std::size_t(y) * strideBytes, rowBytes);
    return true;
  }
}

bool MetalUploadLabels(MetalContext* ctx, void* cmdQueue, const std::int32_t* labels,
                       const std::uint8_t* eDist, const std::uint8_t* eAngle, int width,
                       int height) {
  if (!ctx || !cmdQueue || !labels || width <= 0 || height <= 0) return false;
  @autoreleasepool {
    id<MTLDevice> device = ((__bridge id<MTLCommandQueue>)cmdQueue).device;
    const std::size_t px = std::size_t(width) * std::size_t(height);

    std::lock_guard<std::mutex> lock(ctx->mutex);
    // New buffers rather than overwriting the old ones: a composite from the
    // previous frame may still be reading them, and it holds its own reference.
    ctx->labels = uploadBuffer(device, labels, px * sizeof(std::int32_t));
    // The edge field rides with the labels because it is derived from them.
    if (eDist && eAngle) {
      ctx->edgeDist = uploadBuffer(device, eDist, px);
      ctx->edgeAngle = uploadBuffer(device, eAngle, px);
    } else {
      ctx->edgeDist = nil;
      ctx->edgeAngle = nil;
    }
    ctx->device = device;
    ctx->labelW = width;
    ctx->labelH = height;
    return ctx->labels != nil && (!eDist || !eAngle || (ctx->edgeDist && ctx->edgeAngle));
  }
}

namespace {

/// The body of MetalComposite. Hands back the committed command buffer so the
/// parity entry can wait on exactly this work; the plugin never waits.
id<MTLCommandBuffer> compositeImpl(MetalContext* ctx, void* cmdQueue, const MetalImage& src,
                                   const MetalImage& dst, const ShardDraw* draws, int drawCount,
                                   const std::int32_t* tileStart, int tileCount,
                                   const std::int32_t* tileShardIdx, int tileShardCount,
                                   int tilesX, int tilesY, const CompositeParams& params) {
  if (!ctx || !cmdQueue || !tileStart) return nil;
  if (!fits(src, "source") || !fits(dst, "destination")) return nil;

  @autoreleasepool {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)cmdQueue;
    id<MTLDevice> device = queue.device;
    const Pipelines* p = pipelinesFor(device);
    if (!p) return nil;

    // Once per process: the one line that says the GPU path is really in use,
    // which is otherwise indistinguishable from the CPU path by eye.
    static std::once_flag announced;
    std::call_once(announced, [&] {
      os_log(metalLog(), "Metal render path active on %{public}s (%dx%d)",
             device.name.UTF8String, dst.width, dst.height);
    });

    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->labels || ctx->device != device) return nil;

    // The draw list and the tile arrays are a few hundred kilobytes, so they
    // are uploaded whole every frame. A fresh buffer each time rather than one
    // reused: the previous frame's command buffer may still be reading it.
    id<MTLBuffer> bDraws = uploadBuffer(device, draws, std::size_t(drawCount) * sizeof(ShardDraw));
    id<MTLBuffer> bStart =
        uploadBuffer(device, tileStart, std::size_t(tileCount + 1) * sizeof(std::int32_t));
    id<MTLBuffer> bIdx =
        uploadBuffer(device, tileShardIdx, std::size_t(tileShardCount) * sizeof(std::int32_t));
    if (!bDraws || !bStart || !bIdx) return nil;

    const bool bevelOn = params.bevel > 0.0f && ctx->edgeDist && ctx->edgeAngle;

    // Top-down addressing over bottom-up buffers: start at the last row and
    // step backward, exactly as makeView does on the CPU path.
    ShadeArgs a{};
    a.labelW = ctx->labelW;
    a.labelH = ctx->labelH;
    a.srcW = src.width;
    a.srcH = src.height;
    a.srcTop = int((src.height - 1) * src.rowStrideFloats);
    a.srcStride = -int(src.rowStrideFloats);
    a.dstTop = int((dst.height - 1) * dst.rowStrideFloats);
    a.dstStride = -int(dst.rowStrideFloats);
    a.dstW = dst.width;
    a.dstH = dst.height;
    a.tilesX = tilesX;
    a.tilesY = tilesY;
    a.tileSize = kTileSize;
    a.edgeAA = params.edgeAntialias ? 1 : 0;
    a.showPieces = params.showPieces ? 1 : 0;
    a.bevel = bevelOn ? params.bevel : 0.0f;
    a.bevelWidth = params.bevelWidth;
    a.lightX = params.lightX;
    a.lightY = params.lightY;

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    cb.label = @"ShatterPhysics composite";
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p->shade];
    [enc setBuffer:ctx->labels offset:0 atIndex:0];
    // Something must be bound even with no bevel; the kernel never reads it then.
    [enc setBuffer:bevelOn ? ctx->edgeDist : ctx->labels offset:0 atIndex:1];
    [enc setBuffer:bevelOn ? ctx->edgeAngle : ctx->labels offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)src.buffer offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)dst.buffer offset:0 atIndex:4];
    [enc setBuffer:bDraws offset:0 atIndex:5];
    [enc setBuffer:bStart offset:0 atIndex:6];
    [enc setBuffer:bIdx offset:0 atIndex:7];
    [enc setBytes:&a length:sizeof(a) atIndex:8];
    const MTLSize group = MTLSizeMake(16, 16, 1);
    [enc dispatchThreadgroups:groupsFor(dst.width, dst.height, group)
        threadsPerThreadgroup:group];
    [enc endEncoding];

    // Not waited on: it is ordered on Resolve's own queue ahead of whatever
    // Resolve does with the frame next, which is all the synchronisation needed.
    [cb commit];
    return cb;
  }
}

}  // namespace

bool MetalComposite(MetalContext* ctx, void* cmdQueue, const MetalImage& src,
                    const MetalImage& dst, const ShardDraw* draws, int drawCount,
                    const std::int32_t* tileStart, int tileCount,
                    const std::int32_t* tileShardIdx, int tileShardCount, int tilesX,
                    int tilesY, const CompositeParams& params) {
  @autoreleasepool {
    return compositeImpl(ctx, cmdQueue, src, dst, draws, drawCount, tileStart, tileCount,
                         tileShardIdx, tileShardCount, tilesX, tilesY, params) != nil;
  }
}

bool MetalClear(void* cmdQueue, const MetalImage& dst) {
  if (!cmdQueue || !fits(dst, "destination")) return false;
  @autoreleasepool {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)cmdQueue;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:(__bridge id<MTLBuffer>)dst.buffer
               range:NSMakeRange(0, spanBytes(dst))
               value:0];
    [blit endEncoding];
    [cb commit];
    return true;
  }
}

// ------------------------------------------------------------ parity test entries

namespace {

/// Top-down packed host frame -> bottom-up GPU buffer with padded rows, which is
/// the layout the plugin receives. The padding makes a stride bug visible.
id<MTLBuffer> toGpuLayout(id<MTLDevice> device, const float* hostTopDown, int w, int h,
                          MetalImage& img) {
  const std::ptrdiff_t stride = std::ptrdiff_t(w) * 4 + 12;
  std::vector<float> staged(std::size_t(stride) * std::size_t(h), -7.0f);
  for (int y = 0; y < h; ++y) {
    const float* row = hostTopDown + std::size_t(h - 1 - y) * std::size_t(w) * 4;
    std::memcpy(staged.data() + std::size_t(y) * std::size_t(stride), row,
                std::size_t(w) * 4 * sizeof(float));
  }
  id<MTLBuffer> buf = [device newBufferWithBytes:staged.data()
                                          length:staged.size() * sizeof(float)
                                         options:MTLResourceStorageModeShared];
  img.buffer = (__bridge void*)buf;
  img.width = w;
  img.height = h;
  img.rowStrideFloats = stride;
  return buf;
}

}  // namespace

bool MetalCompositeHostSync(const float* hostSrc, int srcW, int srcH, float* hostDst, int dstW,
                            int dstH, const std::int32_t* labels, const std::uint8_t* eDist,
                            const std::uint8_t* eAngle, int labelW, int labelH,
                            const ShardDraw* draws, int drawCount, const std::int32_t* tileStart,
                            int tileCount, const std::int32_t* tileShardIdx, int tileShardCount,
                            int tilesX, int tilesY, const CompositeParams& params) {
  if (!MetalAvailable()) return false;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    void* q = (__bridge void*)queue;

    MetalImage src, dst;
    id<MTLBuffer> bSrc = toGpuLayout(device, hostSrc, srcW, srcH, src);
    // Poisoned, so a pixel the kernel never wrote shows up as a difference.
    std::vector<float> poison(std::size_t(dstW) * std::size_t(dstH) * 4, -1.0f);
    id<MTLBuffer> bDst = toGpuLayout(device, poison.data(), dstW, dstH, dst);
    if (!bSrc || !bDst) return false;

    MetalContext ctx;
    if (!MetalUploadLabels(&ctx, q, labels, eDist, eAngle, labelW, labelH)) return false;
    id<MTLCommandBuffer> cb = compositeImpl(&ctx, q, src, dst, draws, drawCount, tileStart,
                                            tileCount, tileShardIdx, tileShardCount, tilesX,
                                            tilesY, params);
    if (!cb) return false;
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      logError("parity composite failed", cb.error);
      return false;
    }

    const auto* base = static_cast<const float*>(bDst.contents);
    for (int y = 0; y < dstH; ++y)
      std::memcpy(hostDst + std::size_t(y) * std::size_t(dstW) * 4,
                  base + std::size_t(dstH - 1 - y) * std::size_t(dst.rowStrideFloats),
                  std::size_t(dstW) * 4 * sizeof(float));
    return true;
  }
}

std::uint64_t MetalHashHostSync(const float* hostSrc, int width, int height, float threshold) {
  if (!MetalAvailable()) return 0;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MetalImage src;
    id<MTLBuffer> buf = toGpuLayout(device, hostSrc, width, height, src);
    if (!buf) return 0;
    MetalContext ctx;
    return MetalHashSource(&ctx, (__bridge void*)queue, src, threshold);
  }
}

}  // namespace shp
