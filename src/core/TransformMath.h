// 2D affine transform math, shared verbatim between the CPU compositor and the
// CUDA kernel. Header-only and device-safe on purpose: having two
// implementations of this drift apart is the most likely source of subtle
// CPU/GPU mismatches, so there is exactly one.
//
// Adapted from multi-transform's src/TransformMath.h; MTX_HD became SHP_HD,
// which is defined in Hash.h so that every device-safe header agrees on it.
#pragma once

#include "Hash.h"  // for SHP_HD

namespace shp {

constexpr float kPi = 3.14159265358979323846f;

SHP_HD inline float Lerp(float a, float b, float t) { return a + (b - a) * t; }

SHP_HD inline float Clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

SHP_HD inline float Clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/** @brief A 2D point/vector. */
struct Vec2
{
    float x = 0.0f, y = 0.0f;

    SHP_HD Vec2() {}
    SHP_HD Vec2(float ax, float ay) : x(ax), y(ay) {}
};

SHP_HD inline Vec2 operator+(const Vec2& a, const Vec2& b) { return Vec2(a.x + b.x, a.y + b.y); }
SHP_HD inline Vec2 operator-(const Vec2& a, const Vec2& b) { return Vec2(a.x - b.x, a.y - b.y); }
SHP_HD inline Vec2 operator*(const Vec2& a, float s) { return Vec2(a.x * s, a.y * s); }
SHP_HD inline Vec2 operator*(float s, const Vec2& a) { return Vec2(a.x * s, a.y * s); }
SHP_HD inline Vec2 operator-(const Vec2& a) { return Vec2(-a.x, -a.y); }
SHP_HD inline Vec2& operator+=(Vec2& a, const Vec2& b) { a.x += b.x; a.y += b.y; return a; }
SHP_HD inline Vec2& operator-=(Vec2& a, const Vec2& b) { a.x -= b.x; a.y -= b.y; return a; }

SHP_HD inline float Dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }

/** @brief 2D cross product: the z of the 3D cross. Signed area of the parallelogram. */
SHP_HD inline float Cross(const Vec2& a, const Vec2& b) { return a.x * b.y - a.y * b.x; }

SHP_HD inline Vec2 Perp(const Vec2& a) { return Vec2(-a.y, a.x); }

SHP_HD inline float LengthSq(const Vec2& a) { return a.x * a.x + a.y * a.y; }
SHP_HD inline float Length(const Vec2& a) { return sqrtf(a.x * a.x + a.y * a.y); }

/** @brief Unit vector, or (0,0) for a zero-length input.
 *
 * Returning zero rather than NaN matters: a force applied exactly at a shard's
 * centroid produces a zero direction, and one NaN centroid poisons every
 * subsequent frame of a cached simulation. */
SHP_HD inline Vec2 Normalize(const Vec2& a)
{
    const float lsq = LengthSq(a);
    if (lsq < 1e-20f) return Vec2(0.0f, 0.0f);
    return a * (1.0f / sqrtf(lsq));
}

/** @brief Rotate a vector by an angle in radians. */
SHP_HD inline Vec2 Rotate(const Vec2& v, float radians)
{
    const float c = cosf(radians);
    const float s = sinf(radians);
    return Vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

/** @brief A 2D affine transform stored as the top two rows of a 3x3 matrix.
 *
 *      | m0 m1 m2 |   | x |
 *      | m3 m4 m5 | * | y |
 *      |  0  0  1 |   | 1 |
 */
struct Mat3
{
    float m[6];

    SHP_HD static Mat3 Identity()
    {
        Mat3 r;
        r.m[0] = 1.0f; r.m[1] = 0.0f; r.m[2] = 0.0f;
        r.m[3] = 0.0f; r.m[4] = 1.0f; r.m[5] = 0.0f;
        return r;
    }

    SHP_HD static Mat3 Translate(float tx, float ty)
    {
        Mat3 r = Identity();
        r.m[2] = tx;
        r.m[5] = ty;
        return r;
    }

    SHP_HD static Mat3 Scale(float sx, float sy)
    {
        Mat3 r = Identity();
        r.m[0] = sx;
        r.m[4] = sy;
        return r;
    }

    /** @param radians counter-clockwise rotation.
     *
     * Radians, not degrees as multi-transform's twin uses: every angle in this
     * plugin comes out of the physics, where it is an integrated angular
     * velocity in radians. Converting at each call site invites the one place
     * that forgets. */
    SHP_HD static Mat3 Rotate(float radians)
    {
        const float c = cosf(radians);
        const float s = sinf(radians);
        Mat3 r;
        r.m[0] =  c; r.m[1] = -s; r.m[2] = 0.0f;
        r.m[3] =  s; r.m[4] =  c; r.m[5] = 0.0f;
        return r;
    }

    SHP_HD void Apply(float x, float y, float& outX, float& outY) const
    {
        outX = m[0] * x + m[1] * y + m[2];
        outY = m[3] * x + m[4] * y + m[5];
    }

    SHP_HD Vec2 Apply(const Vec2& p) const
    {
        return Vec2(m[0] * p.x + m[1] * p.y + m[2],
                    m[3] * p.x + m[4] * p.y + m[5]);
    }

    SHP_HD float Determinant() const { return m[0] * m[4] - m[1] * m[3]; }
};

/** @brief Matrix product: applying the result is equivalent to applying @p b then @p a. */
SHP_HD inline Mat3 operator*(const Mat3& a, const Mat3& b)
{
    Mat3 r;
    r.m[0] = a.m[0] * b.m[0] + a.m[1] * b.m[3];
    r.m[1] = a.m[0] * b.m[1] + a.m[1] * b.m[4];
    r.m[2] = a.m[0] * b.m[2] + a.m[1] * b.m[5] + a.m[2];
    r.m[3] = a.m[3] * b.m[0] + a.m[4] * b.m[3];
    r.m[4] = a.m[3] * b.m[1] + a.m[4] * b.m[4];
    r.m[5] = a.m[3] * b.m[2] + a.m[4] * b.m[5] + a.m[5];
    return r;
}

/** @brief Inverse of an affine transform.
 *
 * A degenerate matrix (zero scale) has no inverse; we return identity rather
 * than producing infinities, so a shard shrunk to nothing renders as untransformed
 * instead of NaN garbage. Callers that care should test Determinant() first.
 */
SHP_HD inline Mat3 Invert(const Mat3& t)
{
    const float det = t.Determinant();
    if (det > -1e-12f && det < 1e-12f) return Mat3::Identity();

    const float inv = 1.0f / det;
    Mat3 r;
    r.m[0] =  t.m[4] * inv;
    r.m[1] = -t.m[1] * inv;
    r.m[3] = -t.m[3] * inv;
    r.m[4] =  t.m[0] * inv;
    r.m[2] = -(r.m[0] * t.m[2] + r.m[1] * t.m[5]);
    r.m[5] = -(r.m[3] * t.m[2] + r.m[4] * t.m[5]);
    return r;
}

/** @brief The forward transform taking a shard from its rest position to its posed one.
 *
 * A shard rotates and scales about its own centroid and is then displaced. That
 * pivot choice is not cosmetic: rotating about the frame centre instead would
 * swing distant shards on long arcs, which reads as the whole image spinning
 * rather than as pieces tumbling.
 *
 * @param cx,cy      shard centroid in pixels (the pivot)
 * @param tx,ty      displacement in pixels
 * @param radians    rotation about the centroid
 * @param scale      uniform scale about the centroid
 */
SHP_HD inline Mat3 ShardTransform(float cx, float cy, float tx, float ty,
                                  float radians, float scale)
{
    return Mat3::Translate(cx + tx, cy + ty)
         * Mat3::Rotate(radians)
         * Mat3::Scale(scale, scale)
         * Mat3::Translate(-cx, -cy);
}

}  // namespace shp
