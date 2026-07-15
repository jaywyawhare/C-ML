#!/usr/bin/env bash
set -euo pipefail

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    make -C ./build -j"$(nproc)"
fi

for b in ./build/bin/*; do
    [ -x "$b" ] || continue
    o=$(ASAN_OPTIONS=detect_leaks=1:halt_on_error=0:abort_on_error=0 "$b" 2>&1 || true)
    if grep -q "LeakSanitizer: detected memory leaks" <<<"$o"; then
        n=$(grep -oE 'SUMMARY: AddressSanitizer: [0-9]+ byte\(s\) leaked' <<<"$o" | grep -oE '[0-9]+' | head -1)
        echo "LEAK $(basename "$b"): ${n:-?} bytes"
    else
        echo "OK   $(basename "$b")"
    fi
done
