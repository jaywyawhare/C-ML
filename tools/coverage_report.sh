#!/usr/bin/env bash
# Aggregate per-source branch/line coverage across all test binaries.
#
# Each ctest binary writes its own .gcda recording only what IT executed;
# a proper whole-suite report must merge them per source file. This script
# runs gcov over every .gcda, then merges the annotated listings by source
# path: a line is covered if ANY binary executed it; a branch outcome counts
# as covered if ANY binary took it (SQLite-style union semantics).
#
# Usage: tools/coverage_report.sh [build_dir]
set -euo pipefail
BUILD_DIR="${1:-build-coverage}"

cd "$BUILD_DIR"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

n=0
# Remove stale listings first: old .gcov files anywhere in the tree would be
# merged as if they were current data.
find . -name '*.gcov' -delete
find . -name '*.gcda' | while read -r gcda; do
    d=$(dirname "$gcda")
    out="$TMP/$n"
    mkdir -p "$out"
    # Run gcov from a unique output dir (--object-dir points back at the
    # build tree) so concurrent .gcov names across binaries don't overwrite
    # each other — the merge needs one listing PER BINARY per source.
    gcov -b -c -o "$PWD/$d" "$gcda" > /dev/null 2>&1 || true
    if ls ./*.gcov >/dev/null 2>&1; then
        mv ./*.gcov "$out/" 2>/dev/null || true
    fi
    n=$((n+1))
done

python3 - "$TMP" <<'EOF'
import sys, os, glob, collections

root = sys.argv[1]
# gcov writes .gcov files next to the .gcno/.gcda
covfiles = glob.glob(os.path.join(root, '**', '*.gcov'), recursive=True)

src_lines = collections.defaultdict(dict)     # src -> lineno -> max count
# Branch slots merge by ordinal position within a line ("taken in ANY copy"),
# because several targets instrument the same source (cml / cml_static /
# cml_shared / mock executables); summing would multiply both sides.
src_branches = collections.defaultdict(lambda: collections.defaultdict(
    lambda: {'t': set(), 'k': set()}))        # src -> line -> slot -> sets

for cf in covfiles:
    cur_src = None
    last_line = None
    slot = 0
    with open(cf, errors='replace') as f:
        for raw in f:
            raw = raw.rstrip('\n')
            stripped = raw.lstrip()
            if ':    0:Source:' in raw:
                cur_src = raw.split(':    0:Source:')[1].strip()
                last_line = None
                slot = 0
                continue
            if cur_src is None or '/usr/' in cur_src:
                continue
            # branch/call records have no "count:line:" layout; a branch
            # record belongs to the code line listed just before it.
            if stripped.startswith('branch'):
                if last_line is None:
                    continue
                taken = ('never' not in stripped) and ('taken 0' not in stripped)
                b = src_branches[cur_src][last_line]
                b['t'].add(slot)
                if taken:
                    b['k'].add(slot)
                slot += 1
                continue
            if stripped.startswith('call'):
                continue
            parts = raw.split(':', 2)
            if len(parts) < 3:
                continue
            counter, lno = parts[0], parts[1]
            try:
                lno_i = int(lno)
            except ValueError:
                continue
            if lno_i == 0:
                continue
            counter = counter.strip()
            if counter == '-':
                continue          # non-executable
            elif counter.endswith('*'):
                cnt = int(counter[:-1])
            elif counter in ('#####', '====='):
                cnt = 0           # compiled, never executed by THIS binary
            else:
                try:
                    cnt = int(counter)
                except ValueError:
                    cnt = None
            if cnt is not None:
                prev = src_lines[cur_src].get(lno_i, 0)
                src_lines[cur_src][lno_i] = max(prev, cnt)
                last_line = lno_i

rows = []
for src, lm in src_lines.items():
    executable = [l for l, c in lm.items() if c is not None]
    hit = [l for l in executable if lm[l] > 0]
    bm = src_branches.get(src, {})
    bt = sum(len(b['t']) for b in bm.values())
    bk = sum(len(b['k']) for b in bm.values())
    rows.append((100.0*bk/bt if bt else -1.0, bt-bk, len(executable)-len(hit), src))
rows.sort()

print(f"{'branch%':>8} {'untaken':>8} {'uncov_lines':>11}  file")
for bp, mb, ml, f in rows:
    print(f"{bp:7.1f}% {mb:8d} {ml:11d}  {f.replace('/home/arrry/dev/personal/C-ML-oX/','')}")

tot_bt = sum(r[1] for r in rows)
all_b = sum(sum(len(b['t']) for b in src_branches.get(r[3], {}).values()) for r in rows)
all_k = sum(sum(len(b['k']) for b in src_branches.get(r[3], {}).values()) for r in rows)
tot_l = sum(len([l for l,c in src_lines[r[3]].items()]) for r in rows)
miss_l = sum(r[2] for r in rows)
if all_b:
    print(f"\nTOTAL: line {(tot_l-miss_l)/max(tot_l,1)*100:.1f}% ({tot_l-miss_l}/{tot_l})  "
          f"branch-outcomes-taken {all_k/max(all_b,1)*100:.1f}% ({all_k}/{all_b})")
EOF
