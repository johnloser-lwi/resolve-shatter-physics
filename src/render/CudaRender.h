#pragma once

// The CUDA path, behind an interface with no CUDA headers in it, so the plugin
// translation unit compiles with MSVC and only the .cu goes through nvcc.
//
// Why there is a GPU path at all, when the compute measurements said there was
// no need for one: because those measurements were taken in the wrong place.
//
// Standalone, on 24 threads, this compositor clears 4K in 17.9 ms/frame, of
// which only about 2.4 ms is shard work -- the rest is the bandwidth of clearing
// and walking a 133 MB float frame. On that basis a kernel buys almost nothing.
//
// But Resolve's pipeline is GPU-resident, and a plugin that declines CUDA is not
// merely a little slower: it forces the host to copy every frame down to system
// memory and back up again. At 4K float RGBA that is ~133 MB each way, ~11-22 ms
// per frame of PCIe traffic depending on the slot, serialised against Resolve's
// own GPU work -- and none of it appears in a standalone benchmark. That
// transfer, not the compositing arithmetic, is what stops a CPU plugin playing
// back in realtime.
//
// What stays on the CPU is the segmentation and the physics. The labelling is
// inherently serial and is cached per content change, so it runs once per edit
// rather than once per frame; the solver is well under a millisecond. The trick
// that keeps the CPU-side analysis from costing a download every frame is
// CudaHashSource: the fingerprint is computed on the device and read back as
// eight bytes, so a cache hit transfers almost nothing and the full frame comes
// down only when the content has actually changed.

#include <cstddef>
#include <cstdint>

#include "core/Compositor.h"

namespace shp {

/// True when a usable CUDA device is present. Cached after the first call.
///
/// Checked before declaring CUDA support to the host, because claiming it on a
/// machine with no device would have the host hand over device pointers that
/// cannot be serviced -- and by then there is no way back to the CPU loop, since
/// the pixel addresses are already device memory.
bool CudaAvailable();

/// Per-instance device state: the label image and the per-frame draw list, kept
/// resident between frames.
struct CudaContext;

CudaContext* CudaCreate();
void CudaDestroy(CudaContext* ctx);

/// Content fingerprint, computed on the device.
///
/// Deliberately NOT the same function as the host's hashAlpha. FNV-1a is a
/// sequential chain and reproducing its exact order across thousands of threads
/// would cost more than it saves. This uses an order-independent combine
/// instead, which is all a cache key needs. The two are never compared against
/// each other: a given instance runs on one path or the other, and if a host
/// ever switched mid-session the worst case is one extra re-analysis.
///
/// @param devSrc points at the BOTTOM row with a POSITIVE stride, as the host
///        hands it over -- the top-down flip belongs to the compositing call.
std::uint64_t CudaHashSource(CudaContext* ctx, const float* devSrc, int width, int height,
                             std::ptrdiff_t rowStrideFloats, float threshold, void* stream);

/// Device -> host, so the CPU can segment the frame. Only called when the hash
/// says the content changed. Writes @p hostDst tightly packed, bottom-up,
/// exactly as the device holds it.
bool CudaDownload(CudaContext* ctx, const float* devSrc, int width, int height,
                  std::ptrdiff_t rowStrideFloats, float* hostDst, void* stream);

/// Upload a freshly built label image. Kept resident afterwards, and re-uploaded
/// only when the geometry object changes -- 33 MB at 4K, so once per edit rather
/// than once per frame.
bool CudaUploadLabels(CudaContext* ctx, const std::int32_t* labels, int width, int height,
                      void* stream);

/// Composite into @p devDst using the resident labels.
///
/// @param devSrc,devDst point at the TOP row with a NEGATIVE stride, matching
///        the top-down convention src/core works in.
bool CudaComposite(CudaContext* ctx, const float* devSrc, float* devDst, int width, int height,
                   std::ptrdiff_t srcStrideFloats, std::ptrdiff_t dstStrideFloats,
                   const ShardDraw* draws, int drawCount, const std::int32_t* tileStart,
                   int tileCount, const std::int32_t* tileShardIdx, int tileShardCount,
                   int tilesX, int tilesY, const CompositeParams& params, void* stream);

/// Synchronous whole-chain entry on HOST buffers, for the parity test.
///
/// Exists so the test links the IDENTICAL compiled kernel the plugin does,
/// rather than a second build of "the same" code -- which is precisely the
/// situation a parity test is there to rule out.
bool CudaCompositeHostSync(const float* hostSrc, float* hostDst, const std::int32_t* labels,
                           int width, int height, const ShardDraw* draws, int drawCount,
                           const std::int32_t* tileStart, int tileCount,
                           const std::int32_t* tileShardIdx, int tileShardCount, int tilesX,
                           int tilesY, const CompositeParams& params);

}  // namespace shp
