#!/usr/bin/env bash
# Run every TLA+ model twice: the correct config must pass, and the broken
# config must fail. A spec that cannot catch its own planted bug is worthless,
# so both halves are asserted. Locates tla2tools.jar via $TLA2TOOLS, a cached
# copy, or mise.
set -euo pipefail
cd "$(dirname "$0")/tla"

jar="${TLA2TOOLS:-}"
if [ -z "$jar" ] || [ ! -f "$jar" ]; then
    for c in "$HOME/.cache/tla/tla2tools.jar" /usr/local/lib/tla2tools.jar; do
        [ -f "$c" ] && jar="$c" && break
    done
fi
if [ -z "$jar" ] || [ ! -f "$jar" ]; then
    echo "verification: tla2tools.jar not found; set TLA2TOOLS or install it" >&2
    exit 2
fi

tlc() { java -XX:+UseParallelGC -cp "$jar" tlc2.TLC "$@"; }

MODELS=(IRReset HCQSignal RingAllreduce)
fail=0
for m in "${MODELS[@]}"; do
    echo "=== $m: correct config must PASS ==="
    if tlc -config "${m}_correct.cfg" "${m}.tla" >"/tmp/${m}_c.log" 2>&1; then
        echo "  PASS"
    else
        echo "  UNEXPECTED FAILURE (correct config should hold):"; tail -20 "/tmp/${m}_c.log"; fail=1
    fi
    echo "=== $m: broken config must FAIL (invariant violated) ==="
    if tlc -config "${m}_broken.cfg" "${m}.tla" >"/tmp/${m}_b.log" 2>&1; then
        echo "  UNEXPECTED PASS (broken config should be refuted)"; fail=1
    else
        if grep -q "Invariant .* is violated" "/tmp/${m}_b.log"; then
            echo "  FAIL as expected (counterexample found)"
        else
            echo "  FAILED for the wrong reason:"; tail -20 "/tmp/${m}_b.log"; fail=1
        fi
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "verification: one or more models did not behave as expected" >&2
    exit 1
fi
echo "verification: all models pass-correct / fail-broken as expected"
