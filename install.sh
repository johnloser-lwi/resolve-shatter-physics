#!/usr/bin/env bash
# Install ShatterPhysics.ofx.bundle into the system OFX plugin directory.
#
# Resolve scans for OFX plugins only at launch, so it must be restarted after
# installing, and it holds the bundle open once it has loaded one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$ROOT/build/bundle/ShatterPhysics.ofx.bundle"

case "$(uname -s)" in
  Darwin) DEST="/Library/OFX/Plugins" ;;
  *)      DEST="/usr/OFX/Plugins" ;;
esac
[ "${1:-}" = "--build" ] && "$ROOT/build.sh"

if [ ! -d "$BUNDLE" ]; then
  echo "Bundle not found at $BUNDLE -- run ./build.sh first, or pass --build." >&2
  exit 1
fi

# Probe rather than assume: an OFX directory with relaxed permissions needs no
# sudo, and asking for rights we do not need is just friction.
SUDO=""
if ! mkdir -p "$DEST" 2>/dev/null || [ ! -w "$DEST" ]; then
  SUDO="sudo"
  echo "==> $DEST needs elevated rights"
fi

$SUDO mkdir -p "$DEST"
$SUDO rm -rf "$DEST/ShatterPhysics.ofx.bundle"
$SUDO cp -R "$BUNDLE" "$DEST/"

# Verify, rather than trusting that cp said nothing.
ARCH_DIR="$(ls "$DEST/ShatterPhysics.ofx.bundle/Contents" | grep -v Info.plist | head -1)"
BIN="$DEST/ShatterPhysics.ofx.bundle/Contents/$ARCH_DIR/ShatterPhysics.ofx"
[ -f "$BIN" ] || { echo "Copy succeeded but $BIN is missing." >&2; exit 1; }

echo "    Verified: $BIN"
if command -v file >/dev/null; then echo "    $(file -b "$BIN")"; fi
echo
echo "Installed successfully."
echo
echo "Next:"
echo "  1. Start DaVinci Resolve (it scans for OFX plugins only at launch)."
echo "  2. Put a clip with alpha -- a title, a logo PNG -- on its own track."
echo "  3. Apply OpenFX > Filters > Stylize > 'Shatter Physics'."
echo "  4. Turn on 'Show Pieces' first to confirm the shards are found."
