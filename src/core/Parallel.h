// A parallel-for over scanline bands.
//
// The compositor is a gather -- every destination pixel reads the source and
// the label image and writes only itself -- so splitting it by rows needs no
// locks and no atomics, and produces bit-identical output to the serial version
// regardless of how the rows are distributed. That property is what makes it
// safe to use here at all: the shard loop inside a pixel still runs in the same
// order, so the compositing arithmetic is unchanged.
//
// Deliberately std::thread rather than the host's multi-thread suite: src/core
// may not include an OFX header, and the offline harness has to run this same
// code with no host at all.
#pragma once

#include <algorithm>
#include <thread>
#include <vector>

namespace shp {

/** @brief Run @p fn over [0, count) split into contiguous bands, one per thread.
 *
 * @param fn called as fn(begin, end) with a half-open row range.
 *
 * Falls back to running inline when the work is small. Spawning eight threads to
 * composite a 40-row proxy costs more than it saves, and the threshold matters
 * because the interactive case -- dragging a slider on a small viewer -- is
 * exactly the small case.
 */
template <class Fn>
void ParallelFor(int count, Fn&& fn, int minPerThread = 32) {
  if (count <= 0) return;

  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) hw = 4;

  int threads = std::min(int(hw), (count + minPerThread - 1) / minPerThread);
  if (threads <= 1) {
    fn(0, count);
    return;
  }

  std::vector<std::thread> pool;
  pool.reserve(std::size_t(threads - 1));

  const int band = (count + threads - 1) / threads;
  for (int t = 1; t < threads; ++t) {
    const int begin = t * band;
    if (begin >= count) break;
    const int end = std::min(count, begin + band);
    pool.emplace_back([&fn, begin, end]() { fn(begin, end); });
  }

  // The calling thread takes the first band rather than sitting idle.
  fn(0, std::min(count, band));

  for (std::thread& th : pool) th.join();
}

}  // namespace shp
