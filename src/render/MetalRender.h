#pragma once

// The Metal path, behind an interface with no Objective-C in it, so the plugin
// translation unit stays plain C++ and only MetalRender.mm sees Metal.
//
// It exists for the same reason the CUDA path does -- see CudaRender.h. Resolve
// on macOS is GPU-resident, and a plugin that declines Metal forces it to copy
// every frame off the GPU and back again. What stays on the CPU is identical:
// the segmentation (cached per content change) and the physics.
//
// The one real difference from the CUDA path is that the kernel is NOT the
// shared ShadePixel. CUDA is C++, so the .cu could include core/Compositor.h and
// call the CPU's own function. Metal Shading Language is a separate language
// that cannot include those headers, so MetalKernel.metal is a line-for-line
// port of ShadePixel and its helpers. That breaks the "the kernel contains no
// maths" rule the CUDA path keeps, and is why tests/metalparity.cpp is not a
// formality: it is the only thing holding the two in agreement. A change to
// ShadePixel, StencilHit, BevelShade, SampleBilinear or CompositeOver has to be
// made in MetalKernel.metal as well, and metalparity will fail until it is.
//
// Buffer convention. With Metal enabled, the host's pixel pointer is an
// id<MTLBuffer>, not an address: the CPU must never dereference it, and a byte
// offset is the only arithmetic that makes sense. The buffer starts at the
// BOTTOM row of the image's bounds, with a positive row stride -- the OFX layout,
// untouched. The top-down flip src/core works in happens inside the kernel.

#include <cstddef>
#include <cstdint>

#include "core/Compositor.h"

namespace shp {

/// True when a Metal device is present AND the kernel compiled on it. Cached
/// after the first call.
///
/// Checked before declaring Metal support to the host. Claiming it and then
/// failing to compile would leave every frame black, since by then the pixels
/// are GPU buffers the CPU loop cannot use.
bool MetalAvailable();

/// One image as the host hands it over with Metal enabled.
struct MetalImage {
  void* buffer = nullptr;          ///< id<MTLBuffer>, bridged; never dereferenced on the CPU
  int width = 0, height = 0;
  std::ptrdiff_t rowStrideFloats = 0;  ///< positive; row 0 is the BOTTOM row
};

/// Per-instance GPU state: the label image and edge field, kept resident.
struct MetalContext;

MetalContext* MetalCreate();
void MetalDestroy(MetalContext* ctx);

/// Content fingerprint, computed on the GPU and read back as eight bytes.
/// Same function as CudaHashSource, and for the same reason not the host's
/// hashAlpha. Returns 0 on failure, never otherwise.
std::uint64_t MetalHashSource(MetalContext* ctx, void* cmdQueue, const MetalImage& src,
                              float threshold);

/// GPU -> host, so the CPU can segment the frame. Only called when the hash
/// says the content changed. Writes @p hostDst tightly packed and bottom-up,
/// exactly as the buffer holds it.
bool MetalDownload(MetalContext* ctx, void* cmdQueue, const MetalImage& src, float* hostDst);

/// Upload a freshly built label image and (optionally) the bevel's edge field.
/// Kept resident afterwards and replaced only when the geometry changes.
bool MetalUploadLabels(MetalContext* ctx, void* cmdQueue, const std::int32_t* labels,
                       const std::uint8_t* eDist, const std::uint8_t* eAngle, int width,
                       int height);

/// Composite into @p dst using the resident labels. Encoded on the host's queue
/// and committed without waiting, so it stays ordered with Resolve's own work.
bool MetalComposite(MetalContext* ctx, void* cmdQueue, const MetalImage& src,
                    const MetalImage& dst, const ShardDraw* draws, int drawCount,
                    const std::int32_t* tileStart, int tileCount,
                    const std::int32_t* tileShardIdx, int tileShardCount, int tilesX,
                    int tilesY, const CompositeParams& params);

/// Fill @p dst with transparent black, for frames with nothing to draw. The
/// CPU path's memset is not an option: the "pixels" are a buffer object.
bool MetalClear(void* cmdQueue, const MetalImage& dst);

// ------------------------------------------------------------- parity test entries
//
// Synchronous, on host memory, using the IDENTICAL compiled kernel and the same
// buffer convention the plugin uses: the frames are flipped into bottom-up,
// padded-stride GPU buffers first, so the test also covers the flip and the
// stride rather than only the shading.

/// @param hostSrc,hostDst top-down, tightly packed RGBA float.
/// @param srcW,srcH the source's size; @p dstW,@p dstH the destination's, which
///        differ inside a Fusion comp. The draw list must have been built with
///        the matching FrameLayout.
bool MetalCompositeHostSync(const float* hostSrc, int srcW, int srcH, float* hostDst, int dstW,
                            int dstH, const std::int32_t* labels, const std::uint8_t* eDist,
                            const std::uint8_t* eAngle, int labelW, int labelH,
                            const ShardDraw* draws, int drawCount, const std::int32_t* tileStart,
                            int tileCount, const std::int32_t* tileShardIdx, int tileShardCount,
                            int tilesX, int tilesY, const CompositeParams& params);

/// MetalHashSource on a host frame (top-down, packed). 0 on failure.
std::uint64_t MetalHashHostSync(const float* hostSrc, int width, int height, float threshold);

}  // namespace shp
