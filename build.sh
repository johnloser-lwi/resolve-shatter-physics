#!/usr/bin/env bash
# Builds the plugin, the offline harness and the tests on macOS or Linux.
#
# The Windows counterpart (build.ps1) has to go out of its way to find a CMake
# new enough for the VS toolset. Here the system cmake is fine, so this is the
# plain two-line version it should be.
set -euo pipefail

CONFIG="${1:-Release}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"

if [ "${2:-}" = "--clean" ] || [ "${1:-}" = "--clean" ]; then
  rm -rf "$BUILD"
  [ "${1:-}" = "--clean" ] && CONFIG=Release
fi

if [ ! -f "$ROOT/external/openfx/Support/Library/ofxsCore.cpp" ]; then
  echo "OpenFX SDK missing. Run:" >&2
  echo "  git clone --depth 1 https://github.com/AcademySoftwareFoundation/openfx.git external/openfx" >&2
  exit 1
fi

# On Apple Silicon this produces an arm64 binary, which is what Resolve is.
# Set -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" if a universal build is wanted.
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE="$CONFIG"
cmake --build "$BUILD" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

echo
echo "Built:"
echo "  $BUILD/bundle/ShatterPhysics.ofx.bundle"
[ -f "$BUILD/shattertest" ] && echo "  $BUILD/shattertest"
