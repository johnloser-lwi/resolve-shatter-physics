// Deterministic hashing, shared verbatim between the CPU path and the CUDA
// kernel.
//
// Every random-looking quantity in this plugin -- a shard's initial velocity, its
// spin, its mass jitter, a Voronoi seed's offset -- comes from here, keyed on
// (shardId, seed). Nothing is drawn from a generator with state, because the
// stateless ballistic mode has to produce the same answer when the host renders
// frame 200 first and frame 3 afterwards, and the rigid-body mode has to produce
// a bit-identical run every time its cache is rebuilt.
//
// Adapted from better-line's core/Wiggle.h, and the reason it uses an integer
// avalanche rather than the usual fract(sin(x) * 43758.5453) is worth repeating:
// that idiom's result depends on the platform's sin(), so the CPU and the GPU
// disagree in the low bits and shards land in visibly different places on the
// two paths. An integer hash has no such freedom.
#pragma once

#if defined(__CUDACC__)
    #define SHP_HD __host__ __device__
#else
    #define SHP_HD
    #include <cmath>
#endif

#include <cstdint>

namespace shp {

/** @brief 32-bit integer avalanche (Wang / Jenkins style finalizer). */
SHP_HD inline uint32_t Hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/** @brief Mix two integers into one hash. Order matters; (a,b) != (b,a). */
SHP_HD inline uint32_t Hash32(uint32_t a, uint32_t b)
{
    return Hash32(a ^ (Hash32(b) + 0x9e3779b9u + (a << 6) + (a >> 2)));
}

SHP_HD inline uint32_t Hash32(uint32_t a, uint32_t b, uint32_t c)
{
    return Hash32(Hash32(a, b), c);
}

/** @brief Uniform float in [0,1).
 *
 * 24 bits of mantissa is every bit a float can hold, so taking the top 24 rather
 * than a modulo keeps the distribution flat. */
SHP_HD inline float Hash01(uint32_t x)
{
    return float(Hash32(x) >> 8) * (1.0f / 16777216.0f);
}

SHP_HD inline float Hash01(uint32_t a, uint32_t b)
{
    return float(Hash32(a, b) >> 8) * (1.0f / 16777216.0f);
}

SHP_HD inline float Hash01(uint32_t a, uint32_t b, uint32_t c)
{
    return float(Hash32(a, b, c) >> 8) * (1.0f / 16777216.0f);
}

/** @brief Uniform float in [-1,1]. */
SHP_HD inline float HashSigned(uint32_t a, uint32_t b)
{
    return Hash01(a, b) * 2.0f - 1.0f;
}

SHP_HD inline float HashSigned(uint32_t a, uint32_t b, uint32_t c)
{
    return Hash01(a, b, c) * 2.0f - 1.0f;
}

// Channel tags, so every draw for a given shard is independent.
//
// Named constants rather than 0,1,2,... at the call site: a shard's speed jitter
// and its spin jitter drawing from the same channel would correlate them, and
// that shows up as every fast shard also spinning fast -- subtle enough to look
// like a physics choice rather than a bug.
enum HashChannel : uint32_t {
    kChanVelAngle   = 0x01,
    kChanVelMag     = 0x02,
    kChanSpin       = 0x03,
    kChanSpinJitter = 0x04,
    kChanMass       = 0x05,
    kChanSeedX      = 0x06,
    kChanSeedY      = 0x07,
    kChanTint       = 0x08,
    kChanDetach     = 0x09,
    kChanDepth      = 0x0A,
    kChanTumbleAxis = 0x0B,
    kChanTumbleRate = 0x0C,
};

}  // namespace shp
