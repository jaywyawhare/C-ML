#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

BASELINE_DIR="${CML_REPLAY_BASELINE:-${PROJECT_DIR}/.process_replay/baseline}"
OUTPUT_DIR="${CML_REPLAY_OUTPUT:-/tmp/cml_process_replay_$$}"

usage() {
    echo "Usage: $0 [record|compare|update]"
    echo ""
    echo "  record   Run tests with process replay, save to output dir"
    echo "  compare  Run tests and compare against baseline"
    echo "  update   Run tests and overwrite baseline with new output"
    exit 1
}

build() {
    if [ ! -d "$BUILD_DIR" ]; then
        cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
    fi
    cmake --build "$BUILD_DIR" -j "$(nproc)"
}

run_tests() {
    local dest="$1"
    mkdir -p "$dest"
    export CML_PROCESS_REPLAY=1
    export CML_PROCESS_REPLAY_DIR="$dest"
    cd "$BUILD_DIR"
    ctest --output-on-failure || true
    unset CML_PROCESS_REPLAY
    unset CML_PROCESS_REPLAY_DIR
}

count_kernels() {
    find "$1" -name "*.kernel" 2>/dev/null | wc -l
}

cmd="${1:-compare}"

case "$cmd" in
    record)
        build
        run_tests "$OUTPUT_DIR"
        echo "Recorded $(count_kernels "$OUTPUT_DIR") kernels to $OUTPUT_DIR"
        ;;
    compare)
        if [ ! -d "$BASELINE_DIR" ]; then
            echo "No baseline found at $BASELINE_DIR"
            echo "Run '$0 update' first to create a baseline."
            exit 1
        fi
        build
        run_tests "$OUTPUT_DIR"

        echo "Comparing $(count_kernels "$OUTPUT_DIR") output kernels against $(count_kernels "$BASELINE_DIR") baseline kernels..."

        mismatch=0
        for baseline_file in "$BASELINE_DIR"/*.kernel; do
            [ -f "$baseline_file" ] || continue
            fname="$(basename "$baseline_file")"
            output_file="$OUTPUT_DIR/$fname"
            if [ ! -f "$output_file" ]; then
                echo "MISSING: $fname"
                mismatch=1
            elif ! diff -q "$baseline_file" "$output_file" > /dev/null 2>&1; then
                echo "MISMATCH: $fname"
                diff -u "$baseline_file" "$output_file" || true
                mismatch=1
            fi
        done

        rm -rf "$OUTPUT_DIR"

        if [ "$mismatch" -eq 1 ]; then
            echo "FAIL: Process replay detected codegen regressions."
            exit 1
        fi
        echo "PASS: All kernels match baseline."
        ;;
    update)
        build
        run_tests "$OUTPUT_DIR"
        mkdir -p "$(dirname "$BASELINE_DIR")"
        rm -rf "$BASELINE_DIR"
        mv "$OUTPUT_DIR" "$BASELINE_DIR"
        echo "Baseline updated with $(count_kernels "$BASELINE_DIR") kernels at $BASELINE_DIR"
        ;;
    *)
        usage
        ;;
esac
