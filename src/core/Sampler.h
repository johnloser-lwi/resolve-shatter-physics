// Image sampling, shared between the CPU compositor and the CUDA kernel.
// Device-safe for the same reason as TransformMath.h: one implementation.
//
// Adapted from multi-transform's src/Sampler.h. The difference is that this one
// samples through ConstImageView (which carries a signed row stride, because the
// plugin hands core code a top-down view of a bottom-up OFX buffer by negating
// the stride rather than copying pixels).
#pragma once

#include "ImageView.h"
#include "TransformMath.h"

namespace shp {

enum EdgeMode
{
    kEdgeBlack  = 0,   ///< outside the source is transparent black
    kEdgeClamp  = 1,   ///< outside repeats the border pixel
    kEdgeMirror = 2    ///< outside mirrors back into the image
};

/** @brief A device-side image view.
 *
 * Separate from ConstImageView because a CUDA kernel cannot take a host struct
 * with std:: types in it; this is a plain POD holding a device pointer. On the
 * CPU path it is filled from a ConstImageView and behaves identically. */
struct SampleView
{
    const float* data = nullptr;
    int width = 0;
    int height = 0;
    int rowStrideFloats = 0;   ///< floats per row; may be negative (top-down view)

    SHP_HD const float* Pixel(int x, int y) const
    {
        return data + (long long)(y) * rowStrideFloats + (long long)(x) * 4;
    }
};

inline SampleView MakeSampleView(const ConstImageView& v)
{
    SampleView s;
    s.data = v.data;
    s.width = v.width;
    s.height = v.height;
    s.rowStrideFloats = int(v.rowStride);
    return s;
}

SHP_HD inline int WrapCoord(int v, int n, EdgeMode mode, bool& outside)
{
    if (v >= 0 && v < n) return v;

    outside = true;
    if (mode == kEdgeClamp)
    {
        return v < 0 ? 0 : n - 1;
    }
    if (mode == kEdgeMirror && n > 1)
    {
        const int period = 2 * n - 2;
        int m = v % period;
        if (m < 0) m += period;
        return m < n ? m : period - m;
    }
    return v < 0 ? 0 : n - 1;   // kEdgeBlack: caller discards via `outside`
}

/** @brief Fetch one texel, honouring the edge mode. Writes 4 channels into @p out. */
SHP_HD inline void FetchTexel(const SampleView& img, int x, int y, EdgeMode mode, float* out)
{
    bool outside = false;
    const int cx = WrapCoord(x, img.width,  mode, outside);
    const int cy = WrapCoord(y, img.height, mode, outside);

    if (outside && mode == kEdgeBlack)
    {
        out[0] = out[1] = out[2] = out[3] = 0.0f;
        return;
    }

    const float* p = img.Pixel(cx, cy);
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; out[3] = p[3];
}

/** @brief Bilinear sample at continuous coordinates.
 *
 * @param sx,sy pixel-centre coordinates: the centre of texel (0,0) is (0.5,0.5),
 *              matching how the shard transform maps destination pixel centres.
 */
SHP_HD inline void SampleBilinear(const SampleView& img, float sx, float sy,
                                  EdgeMode mode, float* out)
{
    const float fx = sx - 0.5f;
    const float fy = sy - 0.5f;

    const int x0 = int(floorf(fx));
    const int y0 = int(floorf(fy));
    const float tx = fx - float(x0);
    const float ty = fy - float(y0);

    float p00[4], p10[4], p01[4], p11[4];
    FetchTexel(img, x0,     y0,     mode, p00);
    FetchTexel(img, x0 + 1, y0,     mode, p10);
    FetchTexel(img, x0,     y0 + 1, mode, p01);
    FetchTexel(img, x0 + 1, y0 + 1, mode, p11);

    for (int c = 0; c < 4; ++c)
    {
        const float top    = Lerp(p00[c], p10[c], tx);
        const float bottom = Lerp(p01[c], p11[c], tx);
        out[c] = Lerp(top, bottom, ty);
    }
}

SHP_HD inline void SampleNearest(const SampleView& img, float sx, float sy,
                                 EdgeMode mode, float* out)
{
    FetchTexel(img, int(floorf(sx)), int(floorf(sy)), mode, out);
}

/** @brief Premultiplied "over" composite: @p src over @p dst, in place on @p dst.
 *
 * Resolve hands plugins premultiplied RGBA, so this is the plain Porter-Duff
 * form with no divide. Shards are composited in id order rather than depth
 * order; with opaque source pixels the visible result is the same, and sorting
 * per pixel would cost more than the overlap is worth.
 */
SHP_HD inline void CompositeOver(float* dst, const float* src)
{
    const float ia = 1.0f - src[3];
    dst[0] = src[0] + dst[0] * ia;
    dst[1] = src[1] + dst[1] * ia;
    dst[2] = src[2] + dst[2] * ia;
    dst[3] = src[3] + dst[3] * ia;
}

}  // namespace shp
