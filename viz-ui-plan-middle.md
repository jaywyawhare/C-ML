# VIZ=1 Dashboard UI Polish — Progress (mid-flight)

**Branch:** `viz-ui-check` (worktree at `../C-ML-viz-check`, forked from `master`)
**Scope:** Frontend only (`viz/index.html`, `viz/viz.js`, `viz/viz.css`). **No C code.**
**State:** Implemented + verified locally. **Uncommitted, not pushed, no PR.**

Diffstat:
```
viz/index.html |   7 +
viz/viz.css    | 227 +++++++++++++++++++++++
viz/viz.js     | 357 ++++++++++++++++++++++++++++++++----
3 files changed, 568 insertions(+), 23 deletions(-)
```

---

## Context / why

`VIZ=1 ./example` exports `graph.json` / `training.json` / `model_architecture.json`
/ `kernels.json`, then serves a no-build vanilla-JS dashboard from `viz/`
(`viz/serve.py` on :8001, 3 tabs: Computational Blueprint, Training, Kernel Studio).
It worked but had UI gaps: no liveness feedback, graph wasted the canvas, kernel
panes lacked copy/diff, favicon 404, non-responsive topbar. This pass polishes all
of that without touching the C side (user is separately fixing the crash/degenerate
data).

---

## DONE

### 1. Live status & feedback
- **Connection pill** in topbar (replaced dead 140px spacer at `index.html`):
  green pulsing **Live** / amber **Reconnecting…** / **Idle**, plus ticking
  "updated Ns ago". Wired into SSE `onopen`/`onmessage`/`onerror`
  (`sseConnect`/`sseDisconnect`, `updateConnStatus`, `relativeTime`, 1s ticker).
- **Loading skeletons** (shimmer) for all 3 tabs on first paint, flipping to real
  content or empty state after a `LOAD_GRACE_MS` (1200ms) grace window.
  Helpers: `isLoading`, `skeletonBlock`, `graphSkeleton`, `trainingSkeleton`,
  `codegenSkeleton`; CSS `@keyframes skeleton-shimmer`.

### 2. Graph tab UX
- **Controls overlay** (bottom-right): zoom in / zoom out / fit / re-run layout
  (`buildGraphControls`).
- **Node search** box in header — highlights matches (cyan) + neighborhood, dims
  the rest (`filterGraphNodes`; cy styles `.search-hidden`, `node.search-match`).
- **ResizeObserver** auto-fit on container/window/tab resize (`_cyOpsResizeObserver`).
- **UNKNOWN** ops styled distinctly (italic dashed grey) via `node.unknown`;
  existing dead (red dashed) + fused-cluster (green dashed) still work.

### 3. Kernel Studio
- **Copy** (with "copied" flash) + **Download** buttons per pane, correct file
  extension per backend (`makeCodePane`, `iconBtn`, `downloadText`).
- **Original↔Optimized line diff**: removed lines red-tinted, added lines
  green-tinted (`.code-line.removed` / `.code-line.added`) — makes fusion obvious.
- **Scroll-synced** panes (`syncScroll`).

### 4. Visual polish & responsiveness
- **Favicon** inline SVG data-URI (kills the 404).
- **Tab-switch fade** (`@keyframes tab-fade`).
- **Responsive** `@media` (720/860px): tab labels → icons, pill → dot, legend
  hidden, search narrows.
- **Keyboard shortcuts 1 / 2 / 3** switch tabs (ignored while typing).

### Verification (all passed)
Served updated assets against synthetic fixtures in `/tmp/viz-fixtures`
(40-epoch training, graph with fused/dead/unknown ops, kernels with a real fusion),
captured headlessly (Playwright + vendored Chromium):
- Live pill shows "Live · updated Ns ago"; 4 graph controls; search works;
  4 code-action buttons; **15 removed / 6 added** diff lines; keyboard shortcuts
  (2→Training, 1→Blueprint); narrow-width collapse; graph auto-refit.
- **Zero JS/page errors.** No favicon 404.
- Screenshots: `/tmp/v2_graph.png`, `/tmp/v2_graph_search.png`,
  `/tmp/v2_training.png`, `/tmp/v2_kernels.png`, `/tmp/v2_narrow.png`.
  Baseline (before): `/tmp/viz_dashboard.png`, `/tmp/viz_training.png`,
  `/tmp/viz_kernels.png`.

---

## REMAINING / not done

### Housekeeping (this task)
- [ ] **Commit** the 3 files on `viz-ui-check` (single-line message per repo
  convention). Not committed yet by request.
- [ ] Decide whether to keep the worktree / merge to `master` / open PR
  (explicitly deferred — no push, no PR).
- [ ] Synthetic fixtures live in `/tmp/viz-fixtures` only (test-only, not in repo).

### Not in scope but observed (C side — user is handling)
- `training_loop_example` **segfaults** at end of run (exit 139); only 1 of 100
  epochs recorded in real data.
- Real `training.json` reports `total_params: 0` / empty `layers` (model metadata
  export degenerate).
- Many ops export as **`UNKNOWN`** (op-name mapping gap) — the UI now styles them
  distinctly, but the real fix is in the exporter.
> With real data currently degenerate, the UI was validated against fixtures.
> Re-verify against real output once the C-side export is fixed.

### Possible follow-up UI polish (optional, not started)
- [ ] Training tab: denser metric-card tuning / larger default chart heights
  (only light CSS touched so far).
- [ ] Graph: minimap for large graphs; persist zoom across data refreshes.
- [ ] Kernel Studio: true side-by-side aligned diff (current diff is per-line
  set-membership tint, not line-aligned); syntax-aware diff.
- [ ] Empty/error states: fully unify `.empty-state` vs `.empty-state-box`
  variants (partially consistent now).
- [ ] Accessibility: focus-visible rings, ARIA live-region for the conn pill.

---

## How to re-run / verify locally

```bash
# 1. serve updated assets against fixtures (data dir = CWD, static = viz/ ROOT)
cd /tmp/viz-fixtures
PORT=8001 python3 /home/arrry/dev/personal/C-ML-viz-check/viz/serve.py

# 2. screenshot harness (Playwright via python3.13 + vendored chromium)
python3.13 /tmp/shot_all.py       # 3 tabs + search + diff/action counts
python3.13 /tmp/shot_final.py     # keyboard shortcuts + responsive + console errors
```

Files touched: `viz/index.html`, `viz/viz.js`, `viz/viz.css` (in worktree
`../C-ML-viz-check`).
