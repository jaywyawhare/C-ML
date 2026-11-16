#!/usr/bin/env bash
# Point this clone's hooks at the tracked .githooks directory so the
# clang-format pre-commit gate runs locally. Safe to run repeatedly.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
chmod +x .githooks/* scripts/format.sh scripts/strict-build.sh 2>/dev/null || true
echo "hooks: core.hooksPath -> .githooks (pre-commit clang-format gate active)"
