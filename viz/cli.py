#!/usr/bin/env python3
"""Headless access to the C-ML viz artifacts.

The dashboard serves graph/training/kernels/flamegraph/rewrites over HTTP for a
browser. This reads the same JSON files directly so the trace can be inspected,
filtered, and piped without opening a browser -- useful in CI and over SSH.

    python3 viz/cli.py rewrites                 # every rule firing, in order
    python3 viz/cli.py rewrites --rule fold_mul # only matches of one rule
    python3 viz/cli.py profile --top 10         # hottest kernels
    python3 viz/cli.py graph                     # node/edge/dead/fused summary
    python3 viz/cli.py <cmd> --json | jq ...     # machine-readable

Files are looked up in .cml/ first, then the current directory (same order the
server uses), or from --dir.
"""

import argparse
import json
import os
import sys

DATA_FILES = {
    "graph": "graph.json",
    "training": "training.json",
    "kernels": "kernels.json",
    "flamegraph": "flamegraph.json",
    "rewrites": "rewrites.json",
}


def _load(name, root):
    """Return the parsed artifact, checking .cml/ then the root dir."""
    fname = DATA_FILES[name]
    for path in (os.path.join(root, ".cml", fname), os.path.join(root, fname)):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    sys.exit(f"{fname} not found (run an example with VIZ=1 first)")


def _emit(obj, as_json):
    if as_json:
        json.dump(obj, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return obj


def cmd_rewrites(args):
    d = _load("rewrites", args.dir)
    rows = []
    for p in d.get("passes", []):
        for m in p.get("matches", []):
            if args.rule and args.rule not in m.get("rule", ""):
                continue
            rows.append({"pass": p.get("name", ""), **m})
    if args.json:
        return _emit(rows, True)
    if not rows:
        print("no matching rewrites")
        return
    w = max(len(r["rule"]) for r in rows)
    last = None
    for r in rows:
        if r["pass"] != last:
            last = r["pass"]
            print(f"\n# {last}")
        print(f"  {r['i']:>4}  {r['rule']:<{w}}  {r['op']:<10}  "
              f"{r.get('from','') } -> {r.get('to','')}  {r.get('us',0):.2f}us")


def cmd_profile(args):
    d = _load("flamegraph", args.dir)
    spans = sorted(d.get("spans", []), key=lambda s: s.get("ms", 0), reverse=True)
    top = spans[: args.top]
    if args.json:
        return _emit(top, True)
    total = d.get("total_ms", 0) or 1
    print(f"total {total:.2f} ms across {d.get('num_spans', 0)} executions")
    print(f"  {'kernel':<16}{'kind':<10}{'calls':>8}{'total ms':>12}{'share':>8}")
    for s in top:
        share = 100.0 * s.get("ms", 0) / total
        print(f"  {s.get('op',''):<16}{s.get('kind',''):<10}{s.get('count',0):>8}"
              f"{s.get('ms',0):>12.3f}{share:>7.1f}%")


def cmd_graph(args):
    d = _load("graph", args.dir)
    nodes = d if isinstance(d, dict) else {}
    dead = sum(1 for v in nodes.values() if isinstance(v, dict) and v.get("is_dead"))
    fused = sum(1 for v in nodes.values() if isinstance(v, dict) and v.get("is_fused"))
    summary = {"nodes": len(nodes), "dead": dead, "fused": fused}
    if args.json:
        return _emit(summary, True)
    print(f"nodes {summary['nodes']}  dead {dead}  fused {fused}")


def main():
    ap = argparse.ArgumentParser(prog="cml-viz", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".", help="directory holding the artifacts (default: .)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("rewrites", help="list IR rewrite-rule firings")
    pr.add_argument("--rule", help="only matches whose rule name contains this")
    pr.set_defaults(fn=cmd_rewrites)

    pp = sub.add_parser("profile", help="hottest kernels from the flamegraph")
    pp.add_argument("--top", type=int, default=15, help="how many kernels (default 15)")
    pp.set_defaults(fn=cmd_profile)

    pg = sub.add_parser("graph", help="IR graph node/dead/fused summary")
    pg.set_defaults(fn=cmd_graph)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
