#!/usr/bin/env bash
BIN_DIR="$(cd "$(dirname "$0")/build/bin" && pwd)"
TIMEOUT=5
TMPDIR_RES=$(mktemp -d)
SKIP_LIST="llama_inference mnist_example"
WORKERS=16

run_one() {
    local bin="$1"
    local name
    name=$(basename "$bin")

    if echo "$SKIP_LIST" | grep -qw "$name"; then
        echo "SKIP|$name|0|needs args/data" > "$TMPDIR_RES/$name"
        return
    fi

    output=$(timeout "$TIMEOUT" "$bin" 2>&1); code=$?

    if [[ $code -eq 124 ]]; then
        echo "SLOW|$name|124|timed out >${TIMEOUT}s" > "$TMPDIR_RES/$name"
    elif [[ $code -ne 0 ]]; then
        err=$(echo "$output" | grep -iE "error|fail|assert|fatal" | head -1 | cut -c1-100)
        echo "FAIL|$name|$code|$err" > "$TMPDIR_RES/$name"
    else
        echo "PASS|$name|0|" > "$TMPDIR_RES/$name"
    fi
}

export -f run_one
export TMPDIR_RES TIMEOUT SKIP_LIST

find "$BIN_DIR" -maxdepth 1 -type f -executable -printf '%p\n' | \
    xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}

pass=(); fail=(); slow=(); skip=()

while IFS='|' read -r status name code msg; do
    case $status in
        PASS) pass+=("$name") ;;
        FAIL) fail+=("$name|$code|$msg") ;;
        SLOW) slow+=("$name") ;;
        SKIP) skip+=("$name") ;;
    esac
done < <(cat "$TMPDIR_RES"/* | sort)

rm -rf "$TMPDIR_RES"

echo "================================================"
echo " RESULTS  (timeout=${TIMEOUT}s, ${WORKERS} parallel workers)"
echo "================================================"

echo ""
printf "PASS  [%d]:\n" "${#pass[@]}"
for b in "${pass[@]}"; do printf "  ✓ %s\n" "$b"; done

echo ""
printf "FAIL  [%d]:\n" "${#fail[@]}"
for entry in "${fail[@]}"; do
    IFS='|' read -r b code msg <<< "$entry"
    printf "  ✗ %-42s exit=%-4s %s\n" "$b" "$code" "$msg"
done

echo ""
printf "SLOW / TIMEOUT  [%d]  (>${TIMEOUT}s):\n" "${#slow[@]}"
for b in "${slow[@]}"; do printf "  ~ %s\n" "$b"; done

echo ""
printf "SKIPPED  [%d]:\n" "${#skip[@]}"
for b in "${skip[@]}"; do printf "  - %s\n" "$b"; done

echo ""
echo "Total: $((${#pass[@]} + ${#fail[@]} + ${#slow[@]} + ${#skip[@]})) binaries"
