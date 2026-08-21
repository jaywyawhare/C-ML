#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-$(cd "$(dirname "$0")/build" && pwd)}"

# Leak detection only works if the binaries were built with ASan; against a
# plain build every binary would silently report "OK".
if ! grep -q "ENABLE_SANITIZERS:BOOL=ON" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    echo "error: $BUILD_DIR was not configured with -DENABLE_SANITIZERS=ON" >&2
    echo "hint: BUILD_DIR=$BUILD_DIR cmake -B \"$BUILD_DIR\" -DENABLE_SANITIZERS=ON && cmake --build \"$BUILD_DIR\"" >&2
    exit 1
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    cmake --build "$BUILD_DIR" -j"$(nproc)"
fi

for b in "$BUILD_DIR"/bin/*; do
    [ -x "$b" ] || continue
    o=$(ASAN_OPTIONS=detect_leaks=1:halt_on_error=0:abort_on_error=0 "$b" 2>&1 || true)
    if grep -q "LeakSanitizer: detected memory leaks" <<<"$o"; then
        n=$(grep -oE 'SUMMARY: AddressSanitizer: [0-9]+ byte\(s\) leaked' <<<"$o" | grep -oE '[0-9]+' | head -1)
        echo "LEAK $(basename "$b"): ${n:-?} bytes"
    else
        echo "OK   $(basename "$b")"
    fi
done
