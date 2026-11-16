#!/usr/bin/env bash
# Mirror the CI format-check gate (.github/workflows/ci.yml) locally.
# Usage: scripts/format.sh check   # fail if any tracked C/H file is unformatted
#        scripts/format.sh fix     # reformat every tracked C/H file in place
#        scripts/format.sh check --staged   # only staged files (used by the hook)
set -euo pipefail

REQUIRED_VERSION="20.1.7"

if ! command -v clang-format >/dev/null 2>&1; then
    echo "format: clang-format not found; CI pins clang-format==${REQUIRED_VERSION}" >&2
    echo "format: install with: python -m pip install 'clang-format==${REQUIRED_VERSION}'" >&2
    exit 2
fi

have="$(clang-format --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
if [ "$have" != "$REQUIRED_VERSION" ]; then
    echo "format: clang-format ${have} != pinned ${REQUIRED_VERSION}; output differs by version" >&2
    echo "format: install the pin with: python -m pip install 'clang-format==${REQUIRED_VERSION}'" >&2
    exit 2
fi

cd "$(git rev-parse --show-toplevel)"

mode="${1:-check}"
scope="${2:-all}"

tracked_files() {
    git ls-files 'src/*.c' 'src/**/*.c' 'include/**/*.h' 'tests/*.c' 'tests/**/*.c' 'src/**/*.h' 'tests/**/*.h'
}

staged_files() {
    git diff --cached --name-only --diff-filter=ACMR -- \
        'src/*.c' 'src/**/*.c' 'include/**/*.h' 'tests/*.c' 'tests/**/*.c' 'src/**/*.h' 'tests/**/*.h'
}

if [ "$scope" = "--staged" ]; then
    mapfile -t files < <(staged_files)
else
    mapfile -t files < <(tracked_files)
fi

if [ "${#files[@]}" -eq 0 ]; then
    exit 0
fi

case "$mode" in
check)
    bad=0
    for f in "${files[@]}"; do
        [ -f "$f" ] || continue
        if ! clang-format --dry-run --Werror "$f" >/dev/null 2>&1; then
            echo "not formatted: $f"
            bad=$((bad + 1))
        fi
    done
    if [ "$bad" -ne 0 ]; then
        echo "non-conforming files: $bad" >&2
        echo "fix with: scripts/format.sh fix" >&2
        exit 1
    fi
    ;;
fix)
    for f in "${files[@]}"; do
        [ -f "$f" ] || continue
        clang-format -i "$f"
    done
    ;;
*)
    echo "usage: scripts/format.sh [check|fix] [--staged]" >&2
    exit 2
    ;;
esac
