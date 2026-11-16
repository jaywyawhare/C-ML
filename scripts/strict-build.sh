#!/usr/bin/env bash
# Local pre-push check: build with -Werror under BOTH gcc and clang, because
# CI's warnings-as-errors leg (-DCML_WARNINGS_AS_ERRORS=ON) is the signal that
# otherwise only fires after a push, and gcc vs clang disagree on warnings.
# Usage: scripts/strict-build.sh [gcc|clang]   (default: both)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

build_one() {
    local cc="$1" cxx="$2" dir="$3"
    command -v "$cc" >/dev/null 2>&1 || { echo "strict: $cc not found, skipping"; return 0; }
    echo "=== strict build: $cc ==="
    rm -rf "$dir"
    cmake -S . -B "$dir" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCML_WARNINGS_AS_ERRORS=ON \
        -DCMAKE_C_COMPILER="$cc" \
        -DCMAKE_CXX_COMPILER="$cxx" >/dev/null
    cmake --build "$dir" -j"$(nproc)"
    echo "=== strict build OK: $cc ==="
}

case "${1:-both}" in
gcc)   build_one gcc g++ build-strict-gcc ;;
clang) build_one clang clang++ build-strict-clang ;;
both)
    build_one gcc g++ build-strict-gcc
    build_one clang clang++ build-strict-clang
    ;;
*) echo "usage: scripts/strict-build.sh [gcc|clang|both]" >&2; exit 2 ;;
esac
