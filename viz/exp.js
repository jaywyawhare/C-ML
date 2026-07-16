"use strict";
// C-ML Experiments — W&B-style tracker UI (prototype). Layers 3–10 live here.

const PALETTE = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#38bdf8", "#a855f7", "#ec4899", "#14b8a6", "#eab308", "#f97316"];
const TABS = ["Charts", "Dashboard", "Table", "Overview", "Config", "Diff", "Histograms", "Curves", "System", "Media", "Tables", "Sweep", "Lineage", "Registry", "Launch", "Weave", "Alerts", "Reports", "Logs"];

const S = {
  runs: [],
  color: {},                 // runId -> color
  selected: new Set(),
  tab: "Charts",
  smooth: 0.0,
  logY: false,
  xKey: "step",
  filter: "",                // sidebar search
  tableSort: { key: null, dir: 1 },
  groupBy: "",               // "" = per-run, else config key -> mean±std bands
  reportMd: "# Experiment Report\n\nWrite notes here. Selected runs' final metrics are embedded below.",
  metricFilter: "",          // substring filter for chart panels
  theme: "dark",
  diffA: null, diffB: null,
  starred: new Set(),        // pinned to top
  hidden: new Set(),         // excluded from comparison
};

const $ = (s, r = document) => r.querySelector(s);
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v != null) e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
  return e;
}
const api = (path) => fetch(path).then(r => r.json());

// ── boot ─────────────────────────────────────────────────────────────────
async function boot() {
  document.body.setAttribute("data-theme", S.theme);
  // When embedded in the main viz (iframe), drop our redundant brand/conn chrome
  // so only the tab bar shows under the host's topbar.
  if (window.self !== window.top) document.body.classList.add("embed");
  const tabsEl = $("#tabs");
  tabsEl.innerHTML = "";
  TABS.forEach((t, i) => tabsEl.append(el("div", {
    class: "tab" + (t === S.tab ? " active" : ""), title: i < 9 ? `shortcut: ${i + 1}` : "",
    onclick: () => { S.tab = t; boot(); }
  }, t)));

  if (!window._wired) {
    window._wired = true;
    const tb = el("div", { class: "theme-toggle", title: "Toggle theme (t)",
      onclick: () => { S.theme = S.theme === "dark" ? "light" : "dark"; boot(); } }, "◐");
    $("#conn").before(tb);
    document.addEventListener("keydown", e => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
      const n = parseInt(e.key, 10);
      if (n >= 1 && n <= 9 && TABS[n - 1]) { S.tab = TABS[n - 1]; boot(); }
      else if (e.key === "t") { S.theme = S.theme === "dark" ? "light" : "dark"; boot(); }
    });
  }
  await loadRuns();
  render();
}

async function loadRuns() {
  const runs = await api("/api/runs");
  S.runs = runs;
  runs.forEach((r, i) => { if (!S.color[r.id]) S.color[r.id] = PALETTE[i % PALETTE.length]; });
  if (S.selected.size === 0) runs.forEach(r => S.selected.add(r.id));
  renderSidebar();
}

function runMatches(r, query) {
  if (!query) return true;
  const hay = (r.name + " " + JSON.stringify(r.config) + " " + (r.tags || []).join(" ")).toLowerCase();
  return hay.includes(query.toLowerCase());
}

function renderSidebar() {
  $("#run-count").textContent = S.runs.length;
  const list = $("#run-list");
  list.innerHTML = "";

  const search = el("input", { class: "select", placeholder: "Search runs, config, tags…",
    value: S.filter, style: { width: "100%", marginBottom: "8px" } });
  search.addEventListener("input", () => { S.filter = search.value; applyFilter(); });
  list.append(search);

  const ordered = [...S.runs].sort((a, b) => (S.starred.has(b.id) ? 1 : 0) - (S.starred.has(a.id) ? 1 : 0));
  ordered.forEach(r => {
    const on = S.selected.has(r.id), hidden = S.hidden.has(r.id), star = S.starred.has(r.id);
    const acc = r.summary && (r.summary["val/accuracy"] ?? r.summary["train/accuracy"]);
    const tags = (r.tags || []).map(t => el("span", { class: "tag" }, t));
    const iconBtn = (glyph, title, on2, fn) => el("span", {
      class: "run-act" + (on2 ? " on" : ""), title,
      onclick: e => { e.stopPropagation(); fn(); }
    }, glyph);
    const actions = el("div", { class: "run-acts" },
      iconBtn("★", "Star / pin", star, () => { star ? S.starred.delete(r.id) : S.starred.add(r.id); renderSidebar(); }),
      iconBtn("◍", "Hide from charts", hidden, () => { hidden ? S.hidden.delete(r.id) : S.hidden.add(r.id); renderSidebar(); render(); }),
      iconBtn("▤", "Archive", r.status === "archived", async () => { await postAction("/api/archive", { run: r.id }); await loadRuns(); render(); }),
      iconBtn("✕", "Delete", false, async () => { if (confirm(`Delete run ${r.name}?`)) { S.selected.delete(r.id); await postAction("/api/delete", { run: r.id }); await loadRuns(); render(); } }));
    const item = el("div", { class: "run-item" + (on ? " on" : "") + (hidden ? " hidden-run" : ""), "data-id": r.id, onclick: () => {
      on ? S.selected.delete(r.id) : S.selected.add(r.id); renderSidebar(); render();
    } },
      el("div", { class: "run-swatch", style: { background: on && !hidden ? S.color[r.id] : "#3a3f52" } }),
      el("div", { class: "run-meta" },
        el("div", { class: "run-name" }, star ? "★ " : "", r.name),
        el("div", { class: "run-sub" }, `lr=${r.config.lr ?? "?"}  ${acc != null ? "acc " + (acc * 100).toFixed(1) + "%" : ""}`),
        tags.length ? el("div", { class: "run-tags" }, ...tags) : null,
        actions),
      el("div", { class: "run-status " + (r.status || "") }, r.status || "?"));
    list.append(item);
  });
  applyFilter();
}

function applyFilter() {
  document.querySelectorAll(".run-item").forEach(it => {
    const r = S.runs.find(x => x.id === it.getAttribute("data-id"));
    it.style.display = r && runMatches(r, S.filter) ? "" : "none";
  });
}

const selectedRuns = () => S.runs.filter(r => S.selected.has(r.id) && !S.hidden.has(r.id));
function postAction(path, body) {
  return fetch(path, { method: "POST", body: JSON.stringify(body) }).then(r => r.json());
}

function render() {
  const c = $("#content");
  c.innerHTML = "";
  ({
    Charts: renderCharts, Dashboard: renderDashboard, Table: renderTable, Overview: renderOverview,
    Config: renderConfig, Diff: renderDiff, Histograms: renderHistograms, Curves: renderCurves,
    System: renderSystem, Media: renderMedia, Tables: renderTables, Sweep: renderSweep,
    Lineage: renderLineage, Registry: renderRegistry, Launch: renderLaunch, Weave: renderWeave,
    Alerts: renderAlerts, Reports: renderReports, Logs: renderLogs,
  }[S.tab] || renderCharts)(c);
}

// ── L3 + L5: dynamic comparison panels ─────────────────────────────────────
async function renderCharts(c) {
  const runs = selectedRuns();
  if (!runs.length) return c.append(el("div", { class: "empty" }, "Select one or more runs."));

  const cfgKeys = [...new Set(S.runs.flatMap(r => Object.keys(r.config)))]
    .filter(k => S.runs.every(r => typeof r.config[k] !== "object")).sort();
  const groupSel = el("select", { class: "select", onchange: e => { S.groupBy = e.target.value; render(); } });
  groupSel.append(el("option", { value: "" }, "per run"));
  cfgKeys.forEach(k => { const o = el("option", { value: k }, "group: " + k); if (k === S.groupBy) o.selected = true; groupSel.append(o); });

  const ctrl = el("div", { class: "controls" });
  ctrl.append(
    el("label", {}, "Smoothing",
      el("input", { type: "range", min: "0", max: "0.95", step: "0.05", value: String(S.smooth),
        oninput: e => { S.smooth = +e.target.value; render(); } }),
      el("span", { class: "badge" }, S.smooth.toFixed(2))),
    el("label", { class: "toggle", onclick: () => { S.logY = !S.logY; render(); } },
      el("input", { type: "checkbox", ...(S.logY ? { checked: "" } : {}) }), "log y"),
    el("div", { class: "seg" },
      el("button", { class: S.xKey === "step" ? "on" : "", onclick: () => { S.xKey = "step"; render(); } }, "step"),
      el("button", { class: S.xKey === "wall" ? "on" : "", onclick: () => { S.xKey = "wall"; render(); } }, "wall-time")),
    el("label", {}, "Grouping", groupSel),
    el("label", {}, "Filter",
      el("input", { class: "select", placeholder: "metric substring…", value: S.metricFilter,
        oninput: e => { S.metricFilter = e.target.value; clearTimeout(window._mf); window._mf = setTimeout(render, 250); } })),
  );
  c.append(ctrl);

  let keys = await api("/api/keys");
  if (S.metricFilter) keys = keys.filter(k => k.toLowerCase().includes(S.metricFilter.toLowerCase()));
  if (!keys.length) return c.append(el("div", { class: "empty" }, "No metrics match the filter."));
  const grid = el("div", { class: "panel-grid" });
  c.append(grid);

  if (S.groupBy) {
    // Grouped mean±std bands (W&B "group by").
    const groupColors = {};
    for (const k of keys) {
      const card = el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), k, el("span", { class: "badge", style: { marginLeft: "auto" } }, "grouped by " + S.groupBy)));
      const plot = el("div"); card.append(plot); grid.append(card);
      const gd = await api(`/api/group?key=${encodeURIComponent(k)}&groupby=${encodeURIComponent(S.groupBy)}`);
      const gnames = Object.keys(gd.groups);
      gnames.forEach((g, i) => { if (!groupColors[g]) groupColors[g] = PALETTE[i % PALETTE.length]; });
      bandChart(plot, gd.groups, groupColors, { logY: S.logY });
      card.append(el("div", { class: "legend" }, ...gnames.map(g => el("span", {}, el("i", { style: { background: groupColors[g] } }), `${S.groupBy}=${g}`))));
    }
    return;
  }

  const metrics = await api(`/api/metrics?runs=${runs.map(r => r.id).join(",")}&keys=${keys.map(encodeURIComponent).join(",")}`);
  keys.forEach(k => {
    const card = el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), k));
    const plot = el("div");
    card.append(plot);
    grid.append(card);
    const series = {};
    runs.forEach(r => { const d = (metrics[r.id] || {})[k] || []; if (d.length) series[r.id] = d; });
    lineChart(plot, series, { logY: S.logY, xKey: S.xKey, smooth: S.smooth });
    card.append(legend(Object.keys(series)));
  });
}

// mean±std band chart for grouped runs
function bandChart(container, groups, colors, opts) {
  const W = container.clientWidth || 360, H = 200, m = { t: 10, r: 12, b: 26, l: 44 };
  const names = Object.keys(groups);
  if (!names.length) { container.append(el("div", { class: "empty" }, "no data")); return; }
  const all = names.flatMap(g => groups[g]);
  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const x = d3.scaleLinear().domain(d3.extent(all, d => d.step)).range([m.l, W - m.r]);
  const lo = d3.min(all, d => d.mean - d.std), hi = d3.max(all, d => d.mean + d.std);
  const y = (opts.logY ? d3.scaleLog().clamp(true) : d3.scaleLinear())
    .domain([opts.logY ? Math.max(1e-6, lo) : Math.min(0, lo), hi || 1]).range([H - m.b, m.t]).nice();
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format("~g")));
  names.forEach(g => {
    const s = groups[g], col = colors[g];
    const area = d3.area().x(d => x(d.step)).y0(d => y(Math.max(opts.logY ? 1e-6 : -1e9, d.mean - d.std))).y1(d => y(d.mean + d.std));
    svg.append("path").datum(s).attr("fill", col).attr("opacity", 0.15).attr("d", area);
    const line = d3.line().x(d => x(d.step)).y(d => y(d.mean));
    svg.append("path").datum(s).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2).attr("d", line);
  });
}

function ema(data, alpha, xKey) {
  if (alpha <= 0) return data.map(d => ({ x: d[xKey], y: d.value }));
  let prev = null;
  return data.map(d => { prev = prev == null ? d.value : alpha * prev + (1 - alpha) * d.value; return { x: d[xKey], y: prev }; });
}

function lineChart(container, seriesByRun, opts) {
  const W = container.clientWidth || 360, H = 200, m = { t: 10, r: 12, b: 26, l: 44 };
  const runs = Object.keys(seriesByRun);
  if (!runs.length) { container.append(el("div", { class: "empty" }, "no data")); return; }
  const all = runs.flatMap(rid => ema(seriesByRun[rid], opts.smooth, opts.xKey));
  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const x = d3.scaleLinear().domain(d3.extent(all, d => d.x)).range([m.l, W - m.r]);
  let yVals = all.map(d => d.y);
  let y;
  if (opts.logY) {
    const pos = yVals.filter(v => v > 0);
    y = d3.scaleLog().domain([d3.min(pos) || 1e-6, d3.max(pos) || 1]).range([H - m.b, m.t]).clamp(true);
  } else {
    y = d3.scaleLinear().domain([Math.min(0, d3.min(yVals)), d3.max(yVals) || 1]).nice().range([H - m.b, m.t]);
  }
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format("~g")));
  const line = d3.line().x(d => x(d.x)).y(d => y(d.y)).defined(d => d.y != null && (!opts.logY || d.y > 0));
  runs.forEach(rid => {
    svg.append("path").datum(ema(seriesByRun[rid], opts.smooth, opts.xKey))
      .attr("fill", "none").attr("stroke", S.color[rid]).attr("stroke-width", 1.8).attr("d", line);
  });
}

function legend(runIds) {
  const l = el("div", { class: "legend" });
  runIds.forEach(rid => {
    const r = S.runs.find(x => x.id === rid);
    l.append(el("span", {}, el("i", { style: { background: S.color[rid] } }), r ? r.name : rid));
  });
  return l;
}

// ── L4: config comparison ──────────────────────────────────────────────────
function renderConfig(c) {
  const runs = selectedRuns();
  if (!runs.length) return c.append(el("div", { class: "empty" }, "Select runs to compare configs."));
  const keys = [...new Set(runs.flatMap(r => Object.keys(r.config)))].sort();
  const table = el("table", { class: "grid" });
  const head = el("tr", {}, el("th", {}, "config"));
  runs.forEach(r => head.append(el("th", {}, el("span", { style: { color: S.color[r.id] } }, "● "), r.name)));
  table.append(el("thead", {}, head));
  const body = el("tbody", {});
  keys.forEach(k => {
    const tr = el("tr", {}, el("td", { style: { color: "var(--muted)" } }, k));
    const vals = runs.map(r => r.config[k]);
    const differ = new Set(vals.map(v => JSON.stringify(v))).size > 1;
    runs.forEach((r, i) => tr.append(el("td", { style: differ ? { color: "#f59e0b" } : {} },
      r.config[k] != null ? String(r.config[k]) : "—")));
    body.append(tr);
  });
  table.append(body);
  c.append(el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), "Hyperparameters", el("span", { class: "badge", style: { marginLeft: "auto" } }, "amber = differs")), table));
}

// ── L6: histogram heatmap (distribution over steps) ────────────────────────
async function renderHistograms(c) {
  const runs = selectedRuns();
  if (!runs.length) return c.append(el("div", { class: "empty" }, "Select a run."));
  const run = runs[0];
  const keys = await api(`/api/histkeys?run=${run.id}`);
  if (!keys.length) return c.append(el("div", { class: "empty" }, "No histograms logged for this run."));
  const sel = el("select", { class: "select", onchange: e => draw(e.target.value) });
  keys.forEach(k => sel.append(el("option", {}, k)));
  const bar = el("div", { class: "controls" }, el("label", {}, `Run: ${run.name}`), el("label", {}, "Metric", sel));
  c.append(bar);
  const host = el("div", { class: "card" });
  c.append(host);
  const draw = async (name) => {
    host.innerHTML = "";
    host.append(el("h4", {}, el("span", { class: "accent" }), name, el("span", { class: "badge", style: { marginLeft: "auto" } }, "distribution over training")));
    const data = await api(`/api/histograms?run=${run.id}&name=${encodeURIComponent(name)}`);
    if (!data.length) return;
    data.forEach(d => d.counts = JSON.parse(d.counts));
    const bins = data[0].counts.length;
    const W = host.clientWidth - 28, H = 320, m = { t: 10, r: 12, b: 28, l: 46 };
    const svg = d3.select(host).append("svg").attr("width", W).attr("height", H);
    const x = d3.scaleBand().domain(data.map(d => d.step)).range([m.l, W - m.r]).padding(0.04);
    const globalMin = d3.min(data, d => d.mn), globalMax = d3.max(data, d => d.mx);
    const y = d3.scaleLinear().domain([globalMin, globalMax]).range([H - m.b, m.t]);
    const maxCount = d3.max(data, d => d3.max(d.counts));
    const color = d3.scaleSequential(d3.interpolateInferno).domain([0, Math.log(1 + maxCount)]);
    data.forEach(d => {
      const cw = x.bandwidth();
      const step = (d.mx - d.mn) / bins;
      d.counts.forEach((cnt, bi) => {
        const y0 = d.mn + bi * step, y1 = y0 + step;
        svg.append("rect").attr("x", x(d.step)).attr("width", cw)
          .attr("y", y(y1)).attr("height", Math.max(1, y(y0) - y(y1)))
          .attr("fill", cnt > 0 ? color(Math.log(1 + cnt)) : "#0e1017");
      });
    });
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).tickValues(x.domain().filter((_, i) => i % Math.ceil(data.length / 8) === 0)).tickSizeOuter(0));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(d3.format("~g")));
    host.append(el("div", { class: "legend" }, el("span", {}, "x = training step · y = value · brightness = count (log)")));
  };
  draw(keys[0]);
}

// ── L7: system metrics ─────────────────────────────────────────────────────
async function renderSystem(c) {
  const runs = selectedRuns();
  if (!runs.length) return c.append(el("div", { class: "empty" }, "Select a run."));
  const grid = el("div", { class: "panel-grid" });
  c.append(el("div", { class: "controls" }, el("label", {}, "System utilization over selected runs")), grid);
  for (const metric of [["cpu", "CPU %"], ["rss", "Memory (MB)"]]) {
    const card = el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), metric[1]));
    const plot = el("div"); card.append(plot); grid.append(card);
    const series = {};
    for (const r of runs) {
      const rows = await api(`/api/system?run=${r.id}`);
      if (rows.length) series[r.id] = rows.map(d => ({ step: d.step, wall: d.wall, value: d[metric[0]] }));
    }
    lineChart(plot, series, { logY: false, xKey: "step", smooth: 0 });
    card.append(legend(Object.keys(series)));
  }
}

// ── L8: media (images + tables) ─────────────────────────────────────────────
async function renderMedia(c) {
  const runs = selectedRuns();
  if (!runs.length) return c.append(el("div", { class: "empty" }, "Select a run."));
  const grid = el("div", { class: "panel-grid" });
  c.append(grid);
  for (const r of runs) {
    const items = await api(`/api/media?run=${r.id}`);
    for (const it of items) {
      const card = el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), `${it.name} · ${r.name}`));
      if (it.kind === "image") {
        const cv = el("canvas", { width: it.w, height: it.h, class: "media-img",
          style: { width: Math.min(260, it.w * 3) + "px", height: "auto" } });
        card.append(cv);
        grid.append(card);
        const buf = await fetch(`/api/image?run=${r.id}&path=${encodeURIComponent(it.path)}`).then(x => x.arrayBuffer());
        const rgb = new Uint8Array(buf), img = new ImageData(it.w, it.h);
        for (let p = 0, q = 0; p < rgb.length; p += 3, q += 4) {
          img.data[q] = rgb[p]; img.data[q + 1] = rgb[p + 1]; img.data[q + 2] = rgb[p + 2]; img.data[q + 3] = 255;
        }
        cv.getContext("2d").putImageData(img, 0, 0);
      } else if (it.kind === "table") {
        const rows = it.csv.trim().split("\n").map(l => l.split(","));
        const t = el("table", { class: "grid" });
        t.append(el("thead", {}, el("tr", {}, ...rows[0].map(h => el("th", {}, h)))));
        t.append(el("tbody", {}, ...rows.slice(1).map(r2 => el("tr", {}, ...r2.map(v => el("td", {}, v))))));
        card.append(t);
        grid.append(card);
      }
    }
  }
  if (!grid.children.length) c.append(el("div", { class: "empty" }, "No media logged."));
}

// ── L9: sweep parallel coordinates ─────────────────────────────────────────
async function renderSweep(c) {
  const sweeps = [...new Set(S.runs.map(r => r.sweep).filter(Boolean))];
  if (!sweeps.length) return c.append(el("div", { class: "empty" }, "No sweep runs found."));
  const sel = el("select", { class: "select", onchange: e => draw(e.target.value) });
  sweeps.forEach(s => sel.append(el("option", {}, s)));
  c.append(el("div", { class: "controls" }, el("label", {}, "Sweep", sel), el("label", {}, "Parallel coordinates — each line is a run")));
  const host = el("div", { class: "card" }); c.append(host);
  const draw = async (sid) => {
    host.innerHTML = "";
    const { runs } = await api(`/api/sweep?id=${encodeURIComponent(sid)}`);
    if (!runs.length) return;
    const numeric = k => runs.every(r => r.config[k] != null && !isNaN(+r.config[k]));
    const dims = [...new Set(runs.flatMap(r => Object.keys(r.config)))].filter(numeric);
    const W = host.clientWidth - 28, H = 340, m = { t: 24, r: 30, b: 20, l: 30 };
    const svg = d3.select(host).append("svg").attr("width", W).attr("height", H);
    const xs = d3.scalePoint().domain(dims).range([m.l, W - m.r]);
    const ys = {};
    dims.forEach(k => { ys[k] = d3.scaleLinear().domain(d3.extent(runs, r => +r.config[k])).nice().range([H - m.b, m.t]); });
    dims.forEach(k => {
      const g = svg.append("g").attr("transform", `translate(${xs(k)},0)`);
      g.attr("class", "axis").call(d3.axisLeft(ys[k]).ticks(5).tickFormat(d3.format("~g")));
      g.append("text").attr("y", m.t - 10).attr("text-anchor", "middle").attr("fill", "var(--text)").attr("font-size", "11").text(k);
    });
    const line = d3.line();
    runs.forEach(r => {
      svg.append("path").attr("fill", "none").attr("stroke", S.color[r.id] || "#6366f1")
        .attr("stroke-width", 2).attr("opacity", 0.85)
        .attr("d", line(dims.map(k => [xs(k), ys[k](+r.config[k])])));
    });
    host.append(el("div", { class: "legend" }, ...runs.map(r => el("span", {}, el("i", { style: { background: S.color[r.id] } }), r.name))));

    // Hyperparameter importance (correlation with target metric)
    const imp = await api(`/api/importance?id=${encodeURIComponent(sid)}&target=best_val_accuracy`);
    const ic = el("div", { class: "card", style: { marginTop: "14px" } },
      el("h4", {}, el("span", { class: "accent" }), "Hyperparameter importance",
        el("span", { class: "badge", style: { marginLeft: "auto" } }, "corr. with " + imp.target)));
    if (!imp.importance || !imp.importance.length) {
      ic.append(el("div", { class: "empty" }, imp.note || "not enough runs"));
    } else {
      imp.importance.forEach(row => {
        const mag = Math.abs(row.correlation);
        const bar = el("div", { class: "imp-row" },
          el("span", { class: "imp-name" }, row.param),
          el("div", { class: "imp-track" }, el("div", { class: "imp-fill", style: {
            width: (mag * 100).toFixed(0) + "%", background: row.correlation >= 0 ? "#10b981" : "#ef4444" } })),
          el("span", { class: "imp-val" }, row.correlation.toFixed(3)));
        ic.append(bar);
      });
    }
    host.after(ic);
  };
  draw(sweeps[0]);
}

// ── L10: artifacts ──────────────────────────────────────────────────────────
async function renderArtifacts(c) {
  const runs = selectedRuns();
  if (!runs.length) return c.append(el("div", { class: "empty" }, "Select a run."));
  const table = el("table", { class: "grid" });
  table.append(el("thead", {}, el("tr", {}, ...["Run", "Name", "Type", "Path", "Size", "Hash"].map(h => el("th", {}, h)))));
  const body = el("tbody", {});
  let any = false;
  for (const r of runs) {
    const arts = await api(`/api/artifacts?run=${r.id}`);
    arts.forEach(a => {
      any = true;
      body.append(el("tr", {},
        el("td", {}, el("span", { style: { color: S.color[r.id] } }, "● "), r.name),
        el("td", {}, a.name), el("td", {}, el("span", { class: "badge" }, a.atype)),
        el("td", { style: { color: "var(--muted)" } }, a.path),
        el("td", {}, (a.size / 1024).toFixed(1) + " KB"),
        el("td", { style: { color: "var(--muted)" } }, (a.hash || "").slice(0, 12))));
    });
  }
  table.append(body);
  c.append(el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), "Artifacts & lineage"),
    any ? table : el("div", { class: "empty" }, "No artifacts logged.")));
}

// ── Run table (sortable, W&B-style) ─────────────────────────────────────────
async function renderTable(c) {
  const t = await api("/api/table");
  const runs = t.runs;
  if (!runs.length) return c.append(el("div", { class: "empty" }, "No runs."));
  const cols = [
    { k: "__name", label: "Run", get: r => r.name },
    { k: "status", label: "State", get: r => r.status },
    { k: "__dur", label: "Duration", get: r => r.duration },
    ...t.config_keys.map(k => ({ k: "cfg:" + k, label: k, get: r => r.config[k] })),
    ...t.metric_keys.map(k => ({ k: "met:" + k, label: k, get: r => r.summary[k] })),
  ];
  if (S.tableSort.key) {
    const col = cols.find(x => x.k === S.tableSort.key);
    if (col) runs.sort((a, b) => {
      const va = col.get(a), vb = col.get(b);
      return (va > vb ? 1 : va < vb ? -1 : 0) * S.tableSort.dir;
    });
  }
  const table = el("table", { class: "grid" });
  const head = el("tr", {});
  cols.forEach(col => {
    const arrow = S.tableSort.key === col.k ? (S.tableSort.dir > 0 ? " ▲" : " ▼") : "";
    head.append(el("th", { style: { cursor: "pointer", whiteSpace: "nowrap" }, onclick: () => {
      S.tableSort = { key: col.k, dir: S.tableSort.key === col.k ? -S.tableSort.dir : 1 }; render();
    } }, col.label + arrow));
  });
  table.append(el("thead", {}, head));
  const body = el("tbody", {});
  runs.forEach(r => {
    const tr = el("tr", {});
    cols.forEach(col => {
      let v = col.get(r);
      if (col.k === "__name") return tr.append(el("td", {}, el("span", { style: { color: S.color[r.id] } }, "● "), r.name));
      if (col.k === "status") return tr.append(el("td", {}, el("span", { class: "run-status " + (v || "") }, v || "?")));
      if (col.k === "__dur") return tr.append(el("td", {}, v ? v.toFixed(2) + "s" : "—"));
      if (typeof v === "number") v = Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : (+v).toFixed(4);
      tr.append(el("td", {}, v != null ? String(v) : "—"));
    });
    body.append(tr);
  });
  table.append(body);
  c.append(el("div", { class: "card" },
    el("h4", {}, el("span", { class: "accent" }), "Runs", el("span", { class: "badge", style: { marginLeft: "auto" } }, `${runs.length} runs · click a header to sort`)),
    el("div", { style: { overflowX: "auto" } }, table)));
}

// ── Run overview (metadata + summary + config) ──────────────────────────────
async function renderOverview(c) {
  const runs = selectedRuns();
  if (!runs.length) return c.append(el("div", { class: "empty" }, "Select a run."));
  const grid = el("div", { class: "panel-grid" });
  c.append(grid);
  for (const rr of runs) {
    const r = await api(`/api/overview?run=${rr.id}`);
    const kv = (k, v) => el("div", { class: "kv" }, el("span", { class: "kvk" }, k), el("span", { class: "kvv" }, v));
    const dur = r.duration ? r.duration.toFixed(2) + "s" : "—";
    const started = r.start ? new Date(r.start * 1000).toLocaleString() : "—";
    const nameIn = el("input", { class: "select", value: r.name, style: { width: "100%" } });
    const tagsIn = el("input", { class: "select", value: (r.tags || []).join(", "), placeholder: "tag, tag…", style: { width: "100%" } });
    const notesIn = el("textarea", { class: "select", rows: "2", style: { width: "100%", resize: "vertical" } });
    notesIn.value = r.notes || "";
    const save = el("button", { class: "btn", onclick: async () => {
      await postUpdate(r.id, { name: nameIn.value, tags: tagsIn.value.split(",").map(s => s.trim()).filter(Boolean), notes: notesIn.value });
      await loadRuns(); render();
    } }, "Save");

    const card = el("div", { class: "card" },
      el("h4", {}, el("span", { class: "accent", style: { background: S.color[r.id] } }), r.name,
        el("span", { class: "run-status " + (r.status || ""), style: { marginLeft: "auto" } }, r.status || "?")),
      kv("State", r.status || "?"), kv("Started", started), kv("Duration", dur),
      kv("Host", r.host || "—"), kv("OS", r.os || "—"), kv("Git", r.git || "—"),
      kv("Command", r.cmd || "—"),
      kv("Metrics", `${r.n_metrics} keys · ${r.n_scalars} points`),
      el("div", { class: "sub-h" }, "Summary"),
      ...Object.entries(r.summary).map(([k, v]) => kv(k, typeof v === "number" ? (+v).toFixed(4) : v)),
      el("div", { class: "sub-h" }, "Edit"),
      el("div", { class: "edit-row" }, el("span", { class: "kvk" }, "Name"), nameIn),
      el("div", { class: "edit-row" }, el("span", { class: "kvk" }, "Tags"), tagsIn),
      el("div", { class: "edit-row" }, el("span", { class: "kvk" }, "Notes"), notesIn),
      el("div", { style: { display: "flex", gap: "8px", marginTop: "8px" } }, save,
        el("a", { class: "dl", href: `/api/export?run=${r.id}`, download: `${r.name}.csv`, style: { marginTop: "0", alignSelf: "center" } }, "⤓ Export CSV")));
    grid.append(card);
  }
}

// ── Console logs ─────────────────────────────────────────────────────────────
async function renderLogs(c) {
  const runs = selectedRuns();
  if (!runs.length) return c.append(el("div", { class: "empty" }, "Select a run."));
  const sel = el("select", { class: "select", onchange: e => draw(e.target.value) });
  runs.forEach(r => sel.append(el("option", { value: r.id }, r.name)));
  c.append(el("div", { class: "controls" }, el("label", {}, "Run", sel)));
  const host = el("div", { class: "card" });
  c.append(host);
  const draw = async (rid) => {
    host.innerHTML = "";
    const { text } = await api(`/api/logs?run=${rid}`);
    host.append(el("h4", {}, el("span", { class: "accent" }), "Console output"),
      el("pre", { class: "logs" }, text || "(no console output)"));
  };
  draw(runs[0].id);
}

// ── Model registry (aliased artifacts) ──────────────────────────────────────
async function renderRegistry(c) {
  const reg = await api("/api/registry");
  const table = el("table", { class: "grid" });
  table.append(el("thead", {}, el("tr", {}, ...["Model", "Aliases", "Run", "Size", "Hash"].map(h => el("th", {}, h)))));
  const body = el("tbody", {});
  reg.forEach(a => {
    const aliases = (a.aliases || "").split(",").filter(Boolean).map(x =>
      el("span", { class: "alias" + (x === "best" ? " best" : "") }, x));
    body.append(el("tr", {},
      el("td", {}, a.name), el("td", {}, ...aliases),
      el("td", {}, el("span", { style: { color: S.color[a.run] || "#888" } }, "● "), a.run_name),
      el("td", {}, (a.size / 1024).toFixed(1) + " KB"),
      el("td", { style: { color: "var(--muted)" } }, (a.hash || "").slice(0, 12))));
  });
  table.append(body);
  c.append(el("div", { class: "card" },
    el("h4", {}, el("span", { class: "accent" }), "Model Registry", el("span", { class: "badge", style: { marginLeft: "auto" } }, `${reg.length} versioned artifact(s)`)),
    reg.length ? table : el("div", { class: "empty" }, "No aliased artifacts.")));
}

function postUpdate(run, fields) {
  return fetch("/api/update", { method: "POST", body: JSON.stringify({ run, ...fields }) }).then(r => r.json());
}

// ── Curves (PR / ROC / custom x-y) ──────────────────────────────────────────
async function renderCurves(c) {
  const runs = selectedRuns();
  if (!runs.length) return c.append(el("div", { class: "empty" }, "Select runs."));
  const byName = {};   // curve name -> [{run, xs, ys, xlabel, ylabel}]
  for (const r of runs) {
    const cs = await api(`/api/curves?run=${r.id}`);
    cs.forEach(cur => (byName[cur.name] = byName[cur.name] || []).push({ ...cur, run: r.id }));
  }
  const names = Object.keys(byName);
  if (!names.length) return c.append(el("div", { class: "empty" }, "No curves logged (PR/ROC are logged by hyperband_sweep)."));
  const grid = el("div", { class: "panel-grid" });
  c.append(grid);
  names.forEach(name => {
    const card = el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), name));
    const plot = el("div"); card.append(plot); grid.append(card);
    const series = {};
    byName[name].forEach(cur => series[cur.run] = cur.xs.map((x, i) => ({ x, y: cur.ys[i] })));
    xyChart(plot, series, byName[name][0]);
    card.append(legend(Object.keys(series)));
  });
}

function xyChart(container, seriesByRun, meta) {
  const W = container.clientWidth || 360, H = 210, m = { t: 10, r: 12, b: 30, l: 44 };
  const runs = Object.keys(seriesByRun);
  const all = runs.flatMap(r => seriesByRun[r]);
  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const x = d3.scaleLinear().domain(d3.extent(all, d => d.x)).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([d3.min(all, d => d.y), d3.max(all, d => d.y)]).nice().range([H - m.b, m.t]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
  svg.append("text").attr("x", W / 2).attr("y", H - 4).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", "10").text(meta.xlabel || "x");
  const line = d3.line().x(d => x(d.x)).y(d => y(d.y));
  runs.forEach(r => svg.append("path").datum(seriesByRun[r]).attr("fill", "none").attr("stroke", S.color[r]).attr("stroke-width", 1.8).attr("d", line));
}

// ── Run diff (two runs side by side) ─────────────────────────────────────────
async function renderDiff(c) {
  const runs = S.runs;
  if (runs.length < 2) return c.append(el("div", { class: "empty" }, "Need at least 2 runs."));
  const sel = selfrom => {
    const s = el("select", { class: "select" });
    runs.forEach(r => s.append(el("option", { value: r.id }, r.name)));
    return s;
  };
  S.diffA = S.diffA || runs[0].id;
  S.diffB = S.diffB || runs[1].id;
  const a = sel(), b = sel();
  a.value = S.diffA; b.value = S.diffB;
  a.onchange = () => { S.diffA = a.value; render(); };
  b.onchange = () => { S.diffB = b.value; render(); };
  c.append(el("div", { class: "controls" }, el("label", {}, "A", a), el("label", {}, "B", b)));
  const d = await api(`/api/diff?a=${S.diffA}&b=${S.diffB}`);
  if (d.error) return c.append(el("div", { class: "empty" }, d.error));
  const section = (title, rows) => {
    const table = el("table", { class: "grid" });
    table.append(el("thead", {}, el("tr", {}, el("th", {}, title), el("th", {}, d.a.name), el("th", {}, d.b.name))));
    const body = el("tbody", {});
    rows.forEach(x => body.append(el("tr", { style: x.same ? {} : { background: "rgba(245,158,11,.07)" } },
      el("td", { style: { color: "var(--muted)" } }, x.key),
      el("td", { style: x.same ? {} : { color: "#f59e0b" } }, fmt(x.a)),
      el("td", { style: x.same ? {} : { color: "#f59e0b" } }, fmt(x.b)))));
    table.append(body);
    return el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), title), table);
  };
  const fmt = v => v == null ? "—" : typeof v === "number" ? (+v).toFixed(4) : String(v);
  c.append(el("div", { class: "panel-grid", style: { gridTemplateColumns: "1fr 1fr" } },
    section("Config", d.config_diff), section("Summary", d.summary_diff)));
}

// ── Alerts ──────────────────────────────────────────────────────────────────
async function renderAlerts(c) {
  const alerts = await api("/api/alerts");
  if (!alerts.length) return c.append(el("div", { class: "empty" }, "No alerts 🎉 — all runs healthy."));
  const wrap = el("div", { class: "card" },
    el("h4", {}, el("span", { class: "accent" }), "Alerts", el("span", { class: "badge", style: { marginLeft: "auto" } }, `${alerts.length}`)));
  alerts.forEach(a => {
    wrap.append(el("div", { class: "alert alert-" + (a.level || "info") },
      el("span", { class: "alert-lvl" }, (a.level || "info").toUpperCase()),
      el("span", { class: "alert-run", style: { color: S.color[a.run] || "#aaa" } }, a.run_name || ""),
      el("span", { class: "alert-msg" }, a.message)));
  });
  c.append(wrap);
}

// ── Reports (markdown + embedded final-metric bar chart) ────────────────────
async function renderReports(c) {
  const runs = selectedRuns();
  const ta = el("textarea", { class: "select", rows: "8", style: { width: "100%", resize: "vertical", fontFamily: "ui-monospace, monospace" } });
  ta.value = S.reportMd;
  ta.addEventListener("input", () => { S.reportMd = ta.value; prev.innerHTML = mdToHtml(S.reportMd); });
  const prev = el("div", { class: "report-preview" });
  prev.innerHTML = mdToHtml(S.reportMd);
  c.append(el("div", { class: "panel-grid", style: { gridTemplateColumns: "1fr 1fr" } },
    el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), "Markdown"), ta),
    el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), "Preview"), prev)));

  // Embedded panel: final val/accuracy across selected runs (bar chart)
  const bar = el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), "Final val/accuracy by run"));
  const plot = el("div"); bar.append(plot); c.append(bar);
  const data = runs.map(r => ({ name: r.name, id: r.id, v: (r.summary && (r.summary["val/accuracy"] ?? r.summary["best_val_accuracy"])) || 0 }));
  barChart(plot, data);
}

function barChart(container, data) {
  const W = container.clientWidth || 600, H = Math.max(140, data.length * 34 + 20), m = { t: 8, r: 40, b: 8, l: 160 };
  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const y = d3.scaleBand().domain(data.map(d => d.name)).range([m.t, H - m.b]).padding(0.25);
  const x = d3.scaleLinear().domain([0, d3.max(data, d => d.v) || 1]).range([m.l, W - m.r]);
  data.forEach(d => {
    svg.append("rect").attr("x", m.l).attr("y", y(d.name)).attr("height", y.bandwidth())
      .attr("width", Math.max(0, x(d.v) - m.l)).attr("fill", S.color[d.id] || "#6366f1").attr("rx", 3);
    svg.append("text").attr("x", m.l - 8).attr("y", y(d.name) + y.bandwidth() / 2 + 4)
      .attr("text-anchor", "end").attr("fill", "var(--muted)").attr("font-size", "11").text(d.name);
    svg.append("text").attr("x", x(d.v) + 6).attr("y", y(d.name) + y.bandwidth() / 2 + 4)
      .attr("fill", "var(--text)").attr("font-size", "11").text((d.v * 100).toFixed(1) + "%");
  });
}

function mdToHtml(md) {
  return md.split("\n").map(line => {
    if (/^### /.test(line)) return `<h3>${esc(line.slice(4))}</h3>`;
    if (/^## /.test(line)) return `<h2>${esc(line.slice(3))}</h2>`;
    if (/^# /.test(line)) return `<h1>${esc(line.slice(2))}</h1>`;
    if (/^- /.test(line)) return `<li>${inline(line.slice(2))}</li>`;
    if (line.trim() === "") return "<br>";
    return `<p>${inline(line)}</p>`;
  }).join("");
}
function esc(s) { return s.replace(/[&<>]/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m])); }
function inline(s) { return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>"); }

// ── Dashboard: drag-drop custom panels (persisted to localStorage) ──────────
function dashPanels() { try { return JSON.parse(localStorage.getItem("cml_dash") || "[]"); } catch { return []; } }
function saveDash(p) { localStorage.setItem("cml_dash", JSON.stringify(p)); }

async function renderDashboard(c) {
  const runs = selectedRuns();
  const keys = await api("/api/keys");
  let panels = dashPanels().filter(k => keys.includes(k));
  if (!panels.length) panels = keys.slice(0, 2);

  const addSel = el("select", { class: "select" });
  keys.forEach(k => addSel.append(el("option", {}, k)));
  const ctrl = el("div", { class: "controls" },
    el("label", {}, "Add panel", addSel),
    el("button", { class: "btn", onclick: () => { const p = dashPanels(); if (!p.includes(addSel.value)) { p.push(addSel.value); saveDash(p); render(); } } }, "+ Add"),
    el("span", { class: "badge" }, "drag panels to rearrange · saved to your browser"));
  c.append(ctrl);

  const grid = el("div", { class: "panel-grid dash-grid" });
  c.append(grid);
  if (!runs.length) { grid.append(el("div", { class: "empty" }, "Select runs.")); return; }
  const metrics = await api(`/api/metrics?runs=${runs.map(r => r.id).join(",")}&keys=${panels.map(encodeURIComponent).join(",")}`);
  panels.forEach((k, i) => {
    const card = el("div", { class: "card dash-panel", draggable: "true", "data-i": i });
    card.append(el("h4", {}, el("span", { class: "drag-handle", title: "drag" }, "⠿"), el("span", { class: "accent" }), k,
      el("span", { class: "run-act", style: { marginLeft: "auto" }, title: "remove",
        onclick: () => { const p = dashPanels(); p.splice(p.indexOf(k), 1); saveDash(p); render(); } }, "✕")));
    const plot = el("div"); card.append(plot);
    const series = {}; runs.forEach(r => { const d = (metrics[r.id] || {})[k] || []; if (d.length) series[r.id] = d; });
    lineChart(plot, series, { logY: false, xKey: "step", smooth: S.smooth });
    card.addEventListener("dragstart", e => e.dataTransfer.setData("i", String(i)));
    card.addEventListener("dragover", e => e.preventDefault());
    card.addEventListener("drop", e => { e.preventDefault(); const from = +e.dataTransfer.getData("i"); const p = dashPanels(); const [m] = p.splice(from, 1); p.splice(i, 0, m); saveDash(p); render(); });
    grid.append(card);
  });
}

// ── Launch: job queue + submit ──────────────────────────────────────────────
async function renderLaunch(c) {
  const nameIn = el("input", { class: "select", placeholder: "job name" });
  const cmdIn = el("input", { class: "select", placeholder: "command, e.g. python3 examples/_launch_job.py 0.03", style: { minWidth: "360px" } });
  const submit = el("button", { class: "btn", onclick: async () => {
    if (!cmdIn.value) return;
    await postAction("/api/launch", { name: nameIn.value || undefined, command: cmdIn.value });
    cmdIn.value = ""; render();
  } }, "Submit job");
  c.append(el("div", { class: "controls" }, el("label", {}, "Name", nameIn), el("label", {}, "Command", cmdIn), submit,
    el("span", { class: "badge" }, "run agent: python3 -c 'from cmltrack import launch; launch.run_agent()'")));

  const jobs = await api("/api/jobs");
  const table = el("table", { class: "grid" });
  table.append(el("thead", {}, el("tr", {}, ...["Job", "Status", "Exit", "Command", "Output"].map(h => el("th", {}, h)))));
  const body = el("tbody", {});
  jobs.slice().reverse().forEach(j => body.append(el("tr", {},
    el("td", {}, j.name),
    el("td", {}, el("span", { class: "run-status " + (j.status === "finished" ? "finished" : j.status === "running" ? "running" : "") }, j.status)),
    el("td", {}, j.exit == null ? "—" : String(j.exit)),
    el("td", { style: { color: "var(--muted)", fontFamily: "ui-monospace,monospace", fontSize: "11px" } }, j.command),
    el("td", { style: { color: "var(--muted)", fontSize: "11px", maxWidth: "240px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }, (j.output || "").split("\n").pop()))));
  table.append(body);
  c.append(el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), "Job queue", el("span", { class: "badge", style: { marginLeft: "auto" } }, `${jobs.length} jobs`)),
    jobs.length ? table : el("div", { class: "empty" }, "No jobs — submit one above or run examples/launch_demo.py")));
}

// ── Weave: LLM trace trees ──────────────────────────────────────────────────
async function renderWeave(c) {
  const traces = await api("/api/traces");
  if (!traces.length) return c.append(el("div", { class: "empty" }, "No traces — run examples/weave_demo.py"));
  const list = el("div", { class: "card", style: { flex: "0 0 280px" } }, el("h4", {}, el("span", { class: "accent" }), "Traces"));
  const detail = el("div", { class: "card", style: { flex: "1" } });
  const draw = async (tid) => {
    detail.innerHTML = "";
    const t = await api(`/api/trace?id=${tid}`);
    detail.append(el("h4", {}, el("span", { class: "accent" }), t.name,
      el("span", { class: "badge", style: { marginLeft: "auto" } }, `${t.tokens_in}→${t.tokens_out} tok · $${t.total_cost} · ${t.latency_ms}ms`)));
    // build tree by parent
    const byId = {}; t.spans.forEach(s => byId[s.id] = { ...s, kids: [] });
    const roots = [];
    t.spans.forEach(s => (s.parent && byId[s.parent] ? byId[s.parent].kids : roots).push(byId[s.id]));
    const kindColor = { chain: "#6366f1", llm: "#10b981", tool: "#f59e0b", retriever: "#38bdf8" };
    const walk = (s, depth) => {
      const row = el("div", { class: "span-row", style: { paddingLeft: (depth * 18 + 4) + "px" } },
        el("span", { class: "span-kind", style: { background: (kindColor[s.kind] || "#888") + "33", color: kindColor[s.kind] || "#aaa" } }, s.kind),
        el("span", { class: "span-name" }, s.name),
        s.model ? el("span", { class: "badge" }, s.model) : null,
        s.tokens_in ? el("span", { class: "span-meta" }, `${s.tokens_in}→${s.tokens_out} tok · $${s.cost}`) : null,
        el("span", { class: "span-meta", style: { marginLeft: "auto" } }, `${s.latency_ms}ms`));
      detail.append(row);
      s.kids.forEach(k => walk(k, depth + 1));
    };
    roots.forEach(r => walk(r, 0));
  };
  traces.forEach((t, i) => list.append(el("div", { class: "trace-item", onclick: () => { [...list.querySelectorAll(".trace-item")].forEach(x => x.classList.remove("on")); event.currentTarget.classList.add("on"); draw(t.id); } },
    el("div", { class: "run-name" }, t.name),
    el("div", { class: "run-sub" }, `${t.spans} spans · $${t.total_cost} · ${t.tokens_in + t.tokens_out} tok`))));
  c.append(el("div", { style: { display: "flex", gap: "14px", alignItems: "flex-start" } }, list, detail));
  draw(traces[0].id);
}

// ── W&B Tables: rich tables with media cells ────────────────────────────────
async function renderTables(c) {
  const runs = selectedRuns();
  if (!runs.length) return c.append(el("div", { class: "empty" }, "Select a run."));
  let any = false;
  for (const r of runs) {
    const names = await api(`/api/richtable_list?run=${r.id}`);
    for (const name of names) {
      any = true;
      const t = await api(`/api/richtable?run=${r.id}&name=${encodeURIComponent(name)}`);
      const card = el("div", { class: "card" }, el("h4", {}, el("span", { class: "accent" }), `${name} · ${r.name}`));
      const table = el("table", { class: "grid" });
      let sortCol = null, sortDir = 1;
      const rebuild = () => {
        table.innerHTML = "";
        const rows = t.rows.slice();
        if (sortCol) rows.sort((a, b) => { const x = a[sortCol], y = b[sortCol]; return (x > y ? 1 : x < y ? -1 : 0) * sortDir; });
        table.append(el("thead", {}, el("tr", {}, ...t.columns.map(col => el("th", { style: { cursor: "pointer" },
          onclick: () => { sortDir = sortCol === col ? -sortDir : 1; sortCol = col; rebuild(); } }, col + (sortCol === col ? (sortDir > 0 ? " ▲" : " ▼") : ""))))));
        const body = el("tbody", {});
        rows.forEach(row => {
          const tr = el("tr", {});
          t.columns.forEach(col => {
            const v = row[col];
            if (v && typeof v === "object" && v.img) {
              const cv = el("canvas", { width: v.w, height: v.h, class: "media-img", style: { width: "32px", height: "32px" } });
              tr.append(el("td", {}, cv));
              fetch(`/api/image?run=${r.id}&path=${encodeURIComponent(v.img)}`).then(x => x.arrayBuffer()).then(buf => {
                const rgb = new Uint8Array(buf), im = new ImageData(v.w, v.h);
                for (let p = 0, q = 0; p < rgb.length; p += 3, q += 4) { im.data[q] = rgb[p]; im.data[q + 1] = rgb[p + 1]; im.data[q + 2] = rgb[p + 2]; im.data[q + 3] = 255; }
                cv.getContext("2d").putImageData(im, 0, 0);
              });
            } else tr.append(el("td", {}, v != null ? String(v) : "—"));
          });
          body.append(tr);
        });
        table.append(body);
      };
      rebuild();
      card.append(el("div", { style: { overflowX: "auto" } }, table));
      c.append(card);
    }
  }
  if (!any) c.append(el("div", { class: "empty" }, "No rich tables — run examples/launch_demo.py (logs a predictions table)."));
}

// ── Artifact lineage DAG (cytoscape + dagre) ────────────────────────────────
async function renderLineage(c) {
  const g = await api("/api/lineage");
  if (!g.nodes.length) return c.append(el("div", { class: "empty" }, "No lineage yet — run examples/launch_demo.py (dataset → run → model)."));
  const nRun = g.nodes.filter(n => n.type === "run").length;
  const nArt = g.nodes.filter(n => n.type === "artifact").length;
  const cyDiv = el("div", { style: { width: "100%", height: "72vh", background: "var(--bg)", borderRadius: "8px" } });
  c.append(el("div", { class: "card", style: { padding: "10px" } },
    el("h4", { style: { padding: "2px 4px 8px" } }, el("span", { class: "accent" }), "Artifact lineage",
      el("span", { class: "legend", style: { marginLeft: "12px", gap: "14px" } },
        el("span", {}, el("i", { style: { background: "#10b981", width: "10px", height: "10px", borderRadius: "50%" } }), "artifact"),
        el("span", {}, el("i", { style: { background: "#6366f1", width: "10px", height: "10px", borderRadius: "3px" } }), "run")),
      el("span", { class: "badge", style: { marginLeft: "auto" } }, `${nArt} artifacts · ${nRun} runs · ${g.edges.length} edges`)),
    cyDiv));
  requestAnimationFrame(() => {
    if (typeof cytoscape === "undefined") { cyDiv.append(el("div", { class: "empty" }, "cytoscape not loaded")); return; }
    const cy = cytoscape({
      container: cyDiv,
      elements: [
        ...g.nodes.map(n => ({ data: { id: n.id, label: n.label, type: n.type } })),
        ...g.edges.map(e => ({ data: { source: e.from, target: e.to, label: e.label } })),
      ],
      layout: { name: "dagre", rankDir: "LR", nodeSep: 22, rankSep: 120, edgeSep: 12, ranker: "network-simplex" },
      minZoom: 0.2, maxZoom: 2.5, wheelSensitivity: 0.25,
      style: [
        { selector: "node", style: {
            label: "data(label)", "text-wrap": "wrap", "text-max-width": "150px", "text-valign": "center",
            "text-halign": "center", "font-size": "11px", "font-weight": 600, padding: "10px",
            "border-width": 1, "transition-property": "background-color", } },
        { selector: 'node[type="run"]', style: {
            "background-color": "#20243a", "border-color": "#6366f1", color: "#c7ccf5",
            shape: "round-rectangle", width: "label", height: "26px" } },
        { selector: 'node[type="artifact"]', style: {
            "background-color": "#0e2a20", "border-color": "#10b981", color: "#5eead4",
            shape: "round-rectangle", width: "label", height: "34px", "border-width": 2, "font-size": "12px" } },
        { selector: "edge", style: {
            width: 1.6, "line-color": "#39405c", "target-arrow-color": "#39405c", "target-arrow-shape": "triangle",
            "arrow-scale": 0.9, "curve-style": "bezier", "font-size": "9px", label: "data(label)",
            color: "#8b90a5", "text-background-color": "#0a0b0f", "text-background-opacity": 1,
            "text-background-padding": "2px", "control-point-step-size": 30 } },
        { selector: 'edge[label *= "uses"]', style: { "line-color": "#f59e0b", "target-arrow-color": "#f59e0b", "line-style": "dashed" } },
        { selector: "node:selected", style: { "border-color": "#fff", "border-width": 2 } },
      ],
    });
    cy.on("layoutstop", () => cy.fit(cy.elements(), 40));
    setTimeout(() => cy.fit(cy.elements(), 40), 250);
    cy.on("tap", "node", e => cy.animate({ fit: { eles: e.target.closedNeighborhood(), padding: 60 } }, { duration: 300 }));
  });
}

// live refresh
setInterval(async () => { if (S.tab === "Charts" || S.tab === "Table" || S.tab === "Launch") { await loadRuns(); } }, 4000);
boot();
