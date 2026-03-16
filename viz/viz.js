/* viz.js — C-ML Visualizer (Zero-dependency vanilla JS) */
/* global cytoscape, hljs, VizCharts */

"use strict";

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════
const State = {
  activeTab: "graph",
  graph: null,
  training: null,
  modelArch: null,
  kernels: null,
  // Graph view sub-tab
  graphView: "ops",   // "ops" | "model"
  // CodeGen
  activeAccelerator: "cuda",
  // Training checkboxes
  showTrainingLoss: true,
  showTestingLoss: false,
  showValidationLoss: false,
  showTrainingAcc: true,
  showTestingAcc: false,
  showValidationAcc: false,
  // SSE connections
  _sse: {},
  // Cytoscape instances
  _cyOps: null,
  _cyModel: null,
  // Chart resize observers
  _chartObservers: [],
};

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════
const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => [...(root || document).querySelectorAll(sel)];

function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "style" && typeof v === "object") {
        Object.assign(e.style, v);
      } else if (k.startsWith("on") && typeof v === "function") {
        e.addEventListener(k.slice(2).toLowerCase(), v);
      } else if (k === "className") {
        e.className = v;
      } else if (k === "innerHTML") {
        e.innerHTML = v;
      } else {
        e.setAttribute(k, v);
      }
    }
  }
  for (const c of children) {
    if (typeof c === "string") e.appendChild(document.createTextNode(c));
    else if (c) e.appendChild(c);
  }
  return e;
}

function svgIcon(html, w = 16, h = 16) {
  const span = document.createElement("span");
  span.style.display = "flex";
  span.innerHTML = `<svg width="${w}" height="${h}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${html}</svg>`;
  return span;
}

// ═══════════════════════════════════════════════════════════════
// SSE Manager
// ═══════════════════════════════════════════════════════════════
function sseConnect(name, onData) {
  if (State._sse[name]) State._sse[name].close();

  const es = new EventSource(`/${name}/stream`);
  State._sse[name] = es;

  es.onmessage = (event) => {
    try {
      const json = JSON.parse(event.data);
      if (!json.error) onData(json);
    } catch (e) {
      console.error(`SSE parse error (${name}):`, e);
    }
  };

  es.onerror = () => {
    es.close();
    // Fallback: poll once
    fetch(`/${name}`, { cache: "no-store" })
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d && !d.error) onData(d); })
      .catch(() => {});
    // Retry SSE after 3s
    setTimeout(() => {
      if (State._sse[name] === es) sseConnect(name, onData);
    }, 3000);
  };
}

function sseDisconnect(name) {
  if (State._sse[name]) {
    State._sse[name].close();
    delete State._sse[name];
  }
}

// ═══════════════════════════════════════════════════════════════
// Tab Switching
// ═══════════════════════════════════════════════════════════════
function switchTab(tabId) {
  if (State.activeTab === tabId) return;
  State.activeTab = tabId;

  // Update buttons
  $$(".topbar-button").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === tabId);
  });

  // Update tab content visibility
  $$(".tab-content").forEach(tc => {
    tc.classList.toggle("active", tc.id === `tab-${tabId}`);
  });

  // Manage SSE connections based on active tab
  connectDataForTab(tabId);

  // Resize cytoscape if switching to graph
  if (tabId === "graph" && State._cyOps) {
    requestAnimationFrame(() => {
      State._cyOps.resize();
      State._cyOps.fit(undefined, 40);
    });
  }
}

function connectDataForTab(tabId) {
  const needsGraph    = tabId === "graph";
  const needsTraining = tabId === "graph" || tabId === "training";
  const needsKernels  = tabId === "graph" || tabId === "codegen";

  if (needsGraph) {
    sseConnect("graph", d => { State.graph = d; renderGraphView(); });
    fetch("/model_architecture").then(r => r.ok ? r.json() : null)
      .then(d => { if (d && !d.error) { State.modelArch = d; renderGraphView(); } }).catch(() => {});
  } else {
    sseDisconnect("graph");
  }

  if (needsTraining) {
    sseConnect("training", d => { State.training = d; if (tabId === "training") renderTrainingView(); });
  } else {
    sseDisconnect("training");
  }

  if (needsKernels) {
    sseConnect("kernels", d => { State.kernels = d; if (tabId === "codegen") renderCodeGenView(); });
  } else {
    sseDisconnect("kernels");
  }
}


// ═══════════════════════════════════════════════════════════════
// GRAPH VIEW
// ═══════════════════════════════════════════════════════════════

function getNodeColor(label, isDead, isFused) {
  if (isDead) return "#ef4444";
  if (isFused) return "#10b981";
  if (!label) return "#71717a";
  const name = label.toLowerCase();
  if (name.includes("relu") || name.includes("sigmoid") || name.includes("tanh")) return "#ec4899";
  if (name.includes("conv")) return "#8b5cf6";
  if (name.includes("pool")) return "#06b6d4";
  if (name.includes("loss")) return "#f43f5e";
  return "#64748b";
}

function toCytoscapeElements(graph) {
  if (!graph) return [];
  const elements = [];
  const visibleNodes = new Set();
  const fusedGroups = {};
  const usedAsSource = new Set();

  for (const [id, node] of Object.entries(graph)) {
    visibleNodes.add(String(id));
    const hasValidKernelId = node.fusedKernelId &&
      node.fusedKernelId !== "(nil)" && node.fusedKernelId !== "0x0";
    if (node.is_fused && hasValidKernelId) {
      if (!fusedGroups[node.fusedKernelId]) fusedGroups[node.fusedKernelId] = [];
      fusedGroups[node.fusedKernelId].push(id);
    }
    for (const edge of (node.src || [])) {
      const fromId = Array.isArray(edge) ? edge[1] : edge;
      usedAsSource.add(String(fromId));
    }
  }

  for (const [kernelId, nodeIds] of Object.entries(fusedGroups)) {
    const allDead = nodeIds.every(id => graph[id].is_dead);
    elements.push({
      data: { id: kernelId, label: "", isFusedGroup: true, isDead: allDead },
      classes: "fused-cluster" + (allDead ? " dead" : "")
    });
  }

  for (const [id, node] of Object.entries(graph)) {
    const isDead = node.is_dead;
    const isFused = node.is_fused;
    const label = node.label || String(id);
    const color = node.color || getNodeColor(label, isDead, isFused);
    const width = Math.max(80, label.length * 9 + 40);
    elements.push({
      data: {
        id: String(id), label, color, isDead, isFused, width,
        parent: (isFused && node.fusedKernelId) ? node.fusedKernelId : undefined
      },
      classes: (isDead ? "dead " : "") + (isFused ? "fused" : "")
    });
  }

  for (const [toId, node] of Object.entries(graph)) {
    if (!visibleNodes.has(String(toId))) continue;
    for (const edge of (node.src || [])) {
      let fromId, inputIndex;
      if (Array.isArray(edge)) { inputIndex = edge[0]; fromId = edge[1]; }
      else { fromId = edge; }
      if (!visibleNodes.has(String(fromId))) continue;

      const fromNode = graph[fromId];
      const toNode = graph[toId];
      const isInternal = fromNode.is_fused && toNode.is_fused &&
        fromNode.fusedKernelId && toNode.fusedKernelId &&
        fromNode.fusedKernelId === toNode.fusedKernelId;

      elements.push({
        data: {
          id: `${fromId}->${toId}`,
          source: String(fromId), target: String(toId),
          label: inputIndex !== undefined ? String(inputIndex) : "",
          isInternal
        },
        classes: isInternal ? "internal-edge" : ""
      });
    }
  }
  return elements;
}

function renderGraphView() {
  const container = $("#tab-graph");
  const graph = State.graph;
  const modelSummary = State.modelArch || (State.training && State.training.model_summary) || null;

  // Only rebuild DOM if empty
  if (!container.querySelector(".graph-header")) {
    container.innerHTML = "";

    // Header
    const header = el("div", { className: "graph-header" });

    // View tabs
    const viewTabs = el("div", { className: "view-tabs" });
    const opsTab = el("button", { className: "view-tab active", "data-view": "ops" },
      svgIcon('<line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>', 14, 14),
      "Ops Topology"
    );
    opsTab.appendChild(el("span", { className: "badge", id: "ops-count" }, "0"));
    viewTabs.appendChild(opsTab);

    if (modelSummary) {
      const modelTab = el("button", { className: "view-tab", "data-view": "model" },
        svgIcon('<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>', 14, 14),
        "Architecture"
      );
      viewTabs.appendChild(modelTab);
    }

    header.appendChild(viewTabs);

    // Legend
    const legend = el("div", { className: "legend", id: "ops-legend" });
    [
      { color: "#ef4444", label: "Dead Code", dashed: true },
      { color: "#10b981", label: "Fused Kernel", dashed: true },
      { color: "#f59e0b", label: "Input Flow", dashed: false },
      { color: "#a855f7", label: "Output Flow", dashed: false },
      { color: "#3b82f6", label: "Selected", dashed: false },
    ].forEach(item => {
      const li = el("div", { className: "legend-item" });
      const dot = el("div", { className: "legend-dot" + (item.dashed ? " dashed" : "") });
      if (item.dashed) {
        dot.style.borderColor = item.color;
      } else {
        dot.style.background = item.color;
      }
      li.appendChild(dot);
      li.appendChild(el("span", { className: "legend-label" }, item.label));
      legend.appendChild(li);
    });
    header.appendChild(legend);
    container.appendChild(header);

    // Content area
    const content = el("div", { className: "graph-content" });
    content.appendChild(el("div", { className: "graph-pane", id: "graph-ops-pane", style: { zIndex: "1" } }));
    content.appendChild(el("div", { className: "model-pane", id: "graph-model-pane", style: { display: "none" } }));
    container.appendChild(content);

    // View tab click handlers
    viewTabs.addEventListener("click", (e) => {
      const btn = e.target.closest(".view-tab");
      if (!btn) return;
      const view = btn.dataset.view;
      State.graphView = view;
      viewTabs.querySelectorAll(".view-tab").forEach(b => b.classList.toggle("active", b === btn));
      document.getElementById("graph-ops-pane").style.display = view === "ops" ? "" : "none";
      document.getElementById("graph-ops-pane").style.visibility = view === "ops" ? "visible" : "hidden";
      document.getElementById("graph-model-pane").style.display = view === "model" ? "" : "none";
      document.getElementById("ops-legend").style.display = view === "ops" ? "" : "none";
      if (view === "ops" && State._cyOps) {
        requestAnimationFrame(() => { State._cyOps.resize(); State._cyOps.fit(undefined, 40); });
      }
    });
  }

  // Update node count badge
  const countBadge = $("#ops-count");
  if (countBadge) countBadge.textContent = graph ? Object.keys(graph).length : "0";

  // Render Ops Topology
  const opsPaneEl = document.getElementById("graph-ops-pane");
  if (!graph || Object.keys(graph).length === 0) {
    if (!opsPaneEl.querySelector(".empty-state")) {
      opsPaneEl.innerHTML = "";
      const empty = el("div", { className: "empty-state" },
        svgIcon('<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>', 48, 48),
        el("div", { style: { fontSize: "16px" } }, "Waiting for graph data...")
      );
      opsPaneEl.appendChild(empty);
    }
    return;
  }

  // Clear empty state if present
  const emptyEl = opsPaneEl.querySelector(".empty-state");
  if (emptyEl) { opsPaneEl.innerHTML = ""; }

  // Ensure cytoscape container exists
  if (!opsPaneEl.querySelector("#cy-ops")) {
    opsPaneEl.innerHTML = "";
    opsPaneEl.appendChild(el("div", { id: "cy-ops", style: { width: "100%", height: "100%" } }));
  }

  const elements = toCytoscapeElements(graph);
  const nodeCount = Object.keys(graph).length;
  const baseNodeSep = nodeCount < 10 ? 70 : nodeCount < 20 ? 50 : 35;
  const baseRankSep = nodeCount < 10 ? 90 : nodeCount < 20 ? 65 : 50;
  const spacing = {
    nodeSep: baseNodeSep,
    rankSep: baseRankSep,
    edgeSep: Math.max(25, 50 - nodeCount),
    padding: 35
  };

  if (State._cyOps) {
    State._cyOps.json({ elements });
    State._cyOps.layout({
      name: "dagre", rankDir: "LR", ranker: "network-simplex",
      ...spacing, fit: true, animate: true, animationDuration: 300
    }).run();
    return;
  }

  State._cyOps = cytoscape({
    container: document.getElementById("cy-ops"),
    elements,
    style: cytoscapeOpsStyle(),
    layout: {
      name: "dagre", rankDir: "LR", ranker: "network-simplex",
      ...spacing, fit: true, animate: false
    },
    minZoom: 0.2, maxZoom: 3, autoungrabify: true, boxSelectionEnabled: false
  });

  initCyOpsInteraction(State._cyOps);

  // Model Architecture
  if (modelSummary) {
    renderModelArchitecture(modelSummary);
  }
}

function cytoscapeOpsStyle() {
  return [
    { selector: "node", style: {
      "shape": "round-rectangle", "background-color": "#18181b", "border-color": "data(color)",
      "border-width": 1, "label": "data(label)", "text-wrap": "wrap", "text-max-width": "120px",
      "font-family": "Inter, system-ui, sans-serif", "font-size": 10, "font-weight": 500,
      "color": "#e4e4e7", "text-valign": "center", "text-halign": "center",
      "width": "data(width)", "height": 34, "padding": "5px",
      "transition-property": "background-color, border-width, border-color, width, height",
      "transition-duration": "0.2s", "ghost": "yes", "ghost-offset-x": 0, "ghost-offset-y": 2, "ghost-opacity": 0.1
    }},
    { selector: "node.fused-cluster", style: {
      "background-color": "transparent", "background-opacity": 0, "border-color": "#10b981",
      "border-width": 2, "border-style": "dashed", "label": "", "shape": "round-rectangle", "padding": 12,
      "width": "label", "height": "label"
    }},
    { selector: "node.dead", style: {
      "border-color": "#ef4444", "border-style": "dashed", "border-width": 2,
      "background-color": "#18181b", "color": "#a1a1aa"
    }},
    { selector: "node:selected", style: {
      "border-color": "#fff", "border-width": 2, "background-color": "#27272a", "z-index": 999
    }},
    { selector: "edge", style: {
      "curve-style": "taxi", "taxi-direction": "right", "taxi-turn": 15, "taxi-turn-min-distance": 8,
      "line-color": "#52525b", "line-style": "solid", "width": 1.5,
      "target-arrow-shape": "triangle", "target-arrow-color": "#52525b", "arrow-scale": 0.9,
      "font-size": 9, "color": "#a1a1aa", "text-background-color": "#18181b",
      "text-background-opacity": 1, "text-background-padding": 2, "text-rotation": "autorotate",
      "transition-property": "line-color, width, line-style", "transition-duration": "0.1s"
    }},
    { selector: "edge:selected", style: {
      "line-color": "#fafafa", "target-arrow-color": "#fafafa", "width": 2, "z-index": 999
    }},
    { selector: "node.dimmed", style: { "opacity": 0.75, "z-index": 1, "transition-duration": "0.1s" }},
    { selector: "edge.dimmed", style: { "opacity": 0.75, "z-index": 1, "transition-duration": "0.1s" }},
    { selector: ".highlighted", style: {
      "background-color": "#3b82f6", "line-color": "#3b82f6", "target-arrow-color": "#3b82f6",
      "border-color": "#60a5fa", "border-width": 2, "z-index": 9999, "transition-duration": "0.1s"
    }},
    { selector: "edge.highlighted", style: { "width": 3, "line-style": "solid" }},
    { selector: "edge.incoming-highlighted", style: {
      "line-color": "#f59e0b", "target-arrow-color": "#f59e0b", "width": 3, "line-style": "dashed", "z-index": 9999
    }},
    { selector: "edge.outgoing-highlighted", style: {
      "line-color": "#a855f7", "target-arrow-color": "#a855f7", "width": 3, "line-style": "solid", "z-index": 9999
    }},
    { selector: "node.incoming-highlighted", style: { "border-color": "#f59e0b", "border-width": 3 }},
    { selector: "node.outgoing-highlighted", style: { "border-color": "#a855f7", "border-width": 3 }},
    { selector: ".internal-edge", style: {
      "line-color": "#71717a", "target-arrow-color": "#71717a", "width": 1.5, "line-style": "solid", "z-index": 10
    }}
  ];
}

function initCyOpsInteraction(cy) {
  const resetHighlights = () => {
    cy.elements().removeClass("highlighted incoming-highlighted outgoing-highlighted dimmed");
  };

  const highlightNode = (node) => {
    node.addClass("highlighted").removeClass("dimmed");
    if (node.isChild()) node.parent().removeClass("dimmed");
  };

  cy.on("tap", (evt) => {
    if (evt.target === cy) { resetHighlights(); return; }
    const target = evt.target;
    resetHighlights();
    cy.elements().addClass("dimmed");

    if (target.isEdge()) {
      target.addClass("highlighted").removeClass("dimmed");
      highlightNode(target.source());
      highlightNode(target.target());
    } else if (target.isNode()) {
      highlightNode(target);
      const incomers = target.incomers("edge");
      incomers.addClass("incoming-highlighted").removeClass("dimmed");
      incomers.sources().forEach(s => { s.addClass("incoming-highlighted").removeClass("dimmed"); if (s.isChild()) s.parent().removeClass("dimmed"); });
      const outgoers = target.outgoers("edge");
      outgoers.addClass("outgoing-highlighted").removeClass("dimmed");
      outgoers.targets().forEach(t => { t.addClass("outgoing-highlighted").removeClass("dimmed"); if (t.isChild()) t.parent().removeClass("dimmed"); });
    }
  });

  cy.on("tap", "node", (evt) => {
    cy.animate({ center: { eles: evt.target }, zoom: 1.2 }, { duration: 300 });
  });
}


// ═══════════════════════════════════════════════════════════════
// MODEL ARCHITECTURE VIEW
// ═══════════════════════════════════════════════════════════════

function parseModelSummary(summary) {
  if (!summary || summary === "N/A") return [];
  try {
    const parsed = typeof summary === "string" ? JSON.parse(summary) : summary;
    if (parsed && parsed.layers && Array.isArray(parsed.layers)) {
      return parsed.layers.map(layer => ({
        name: layer.type || layer.name || "Unknown",
        params: {
          in_features: layer.in_features, out_features: layer.out_features,
          in_channels: layer.in_channels, out_channels: layer.out_channels,
          kernel_size: layer.kernel_size, stride: layer.stride, padding: layer.padding,
          has_bias: layer.has_bias, num_params: layer.num_params,
        }
      }));
    }
  } catch (e) { /* not JSON */ }

  const layers = [];
  let current = typeof summary === "string" ? summary.trim() : String(summary);
  if (current.startsWith("Sequential(")) current = current.slice(11, -1);

  let depth = 0, start = 0;
  for (let i = 0; i < current.length; i++) {
    if (current[i] === "(") depth++;
    else if (current[i] === ")") depth--;
    else if (current[i] === "," && depth === 0) {
      const s = current.slice(start, i).trim();
      if (s) layers.push(parseLayer(s));
      start = i + 1;
    }
  }
  const last = current.slice(start).trim();
  if (last) layers.push(parseLayer(last));
  return layers;
}

function parseLayer(layerStr) {
  layerStr = layerStr.trim();
  const parenIdx = layerStr.indexOf("(");
  const name = parenIdx > 0 ? layerStr.slice(0, parenIdx).trim() : layerStr;
  const params = {};
  if (parenIdx > 0) {
    const ps = layerStr.slice(parenIdx + 1, -1);
    const arrow = ps.match(/(\d+)\s*->\s*(\d+)/);
    if (arrow) { params.inputSize = parseInt(arrow[1]); params.outputSize = parseInt(arrow[2]); }
    else {
      const nums = ps.match(/\d+/g);
      if (nums && nums.length >= 2) { params.inputSize = parseInt(nums[0]); params.outputSize = parseInt(nums[1]); }
      else if (nums && nums.length === 1) { params.size = parseInt(nums[0]); }
    }
  }
  return { name, params };
}

function getLayerColor(name) {
  const n = name.toLowerCase();
  if (n.includes("relu")) return "#ec4899";
  if (n.includes("linear")) return "#3b82f6";
  if (n.includes("conv")) return "#8b5cf6";
  if (n.includes("pool") || n.includes("lrn")) return "#14b8a6";
  if (n.includes("concat")) return "#f59e0b";
  if (n.includes("tanh")) return "#a855f7";
  if (n.includes("sigmoid")) return "#f472b6";
  if (n.includes("softmax")) return "#eab308";
  if (n.includes("batchnorm") || n.includes("layernorm")) return "#06b6d4";
  if (n.includes("dropout")) return "#f43f5e";
  return "#64748b";
}

function modelToCytoscapeElements(layers) {
  if (!layers || layers.length === 0) return [];
  const elements = [];

  elements.push({ data: { id: "input", label: "Input", color: "#10b981", type: "io" } });

  layers.forEach((layer, idx) => {
    const color = getLayerColor(layer.name);
    let details = "";
    if (layer.params.in_features && layer.params.out_features) {
      details = `${layer.params.in_features} → ${layer.params.out_features}`;
    } else if (layer.params.in_channels && layer.params.out_channels) {
      details = `${layer.params.in_channels}ch → ${layer.params.out_channels}ch`;
      if (layer.params.kernel_size) details += ` (k${layer.params.kernel_size})`;
    } else if (layer.params.inputSize && layer.params.outputSize) {
      details = `${layer.params.inputSize} → ${layer.params.outputSize}`;
    } else if (layer.params.size) {
      details = `size: ${layer.params.size}`;
    } else if (layer.params.kernel_size) {
      details = `kernel: ${layer.params.kernel_size}`;
    }
    if (layer.params.num_params && layer.params.num_params > 0) {
      const ps = layer.params.num_params >= 1000 ? `${(layer.params.num_params / 1000).toFixed(1)}K` : String(layer.params.num_params);
      details = details ? `${details} (${ps} params)` : `${ps} params`;
    }
    const fullLabel = details ? `${layer.name}\n${details}` : layer.name;
    elements.push({ data: { id: `layer-${idx}`, label: fullLabel, color, type: "layer" } });
    elements.push({ data: { id: `${idx === 0 ? "input" : `layer-${idx - 1}`}->layer-${idx}`, source: idx === 0 ? "input" : `layer-${idx - 1}`, target: `layer-${idx}` } });
  });

  elements.push({ data: { id: "output", label: "Output", color: "#10b981", type: "io" } });
  elements.push({ data: { id: `layer-${layers.length - 1}->output`, source: `layer-${layers.length - 1}`, target: "output" } });
  return elements;
}

function renderModelArchitecture(modelSummary) {
  const pane = document.getElementById("graph-model-pane");
  if (!pane) return;
  const layers = parseModelSummary(modelSummary);
  if (layers.length === 0) {
    pane.innerHTML = '<div style="padding:24px;text-align:center;color:var(--muted)">No model architecture available</div>';
    return;
  }

  pane.innerHTML = '<div id="cy-model" style="width:100%;height:100%"></div>';
  const elements = modelToCytoscapeElements(layers);

  if (State._cyModel) { State._cyModel.destroy(); State._cyModel = null; }

  State._cyModel = cytoscape({
    container: document.getElementById("cy-model"),
    elements,
    style: [
      { selector: 'node[type="layer"]', style: {
        "shape": "round-rectangle", "background-color": "transparent", "background-opacity": 0,
        "border-color": "data(color)", "border-width": 3, "label": "data(label)",
        "text-wrap": "wrap", "text-max-width": "280px",
        "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace", "font-size": 26,
        "color": "#fff", "text-valign": "center", "text-halign": "center",
        "width": 320, "height": 90, "padding": "20px 28px"
      }},
      { selector: 'node[type="io"]', style: {
        "shape": "round-rectangle", "background-color": "transparent", "background-opacity": 0,
        "border-color": "data(color)", "border-width": 3, "label": "data(label)",
        "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace", "font-size": 26,
        "color": "#f9fafb", "text-valign": "center", "text-halign": "center",
        "width": 320, "height": 90, "padding": "20px 28px"
      }},
      { selector: "edge", style: {
        "curve-style": "bezier", "line-color": "#94a3b8", "target-arrow-color": "#94a3b8",
        "target-arrow-shape": "triangle", "width": 2.5, "opacity": 0.7
      }}
    ],
    layout: {
      name: "elk",
      elk: {
        algorithm: "layered", "elk.direction": "RIGHT", "elk.edgeRouting": "ORTHOGONAL",
        "spacing.nodeNodeBetweenLayers": "50", "spacing.edgeEdgeBetweenLayers": "20",
        "spacing.nodeNode": "30", "layered.spacing.nodeNodeBetweenLayers": "50",
        "layered.spacing.edgeNodeBetweenLayers": "20", "layered.spacing.edgeEdgeBetweenLayers": "15",
        "elk.layered.nodePlacement.strategy": "SIMPLE",
        "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
        "elk.layered.cycleBreaking.strategy": "GREEDY",
        "elk.layered.compaction.postCompaction": "true",
        "elk.layered.nodePlacement.bk.fixedAlignment": "LEFTUP",
        "elk.padding": "[top=40,left=40,bottom=40,right=40]"
      },
      nodeDimensionsIncludeLabels: true, fit: true, padding: 40
    },
    minZoom: 0.1, maxZoom: 3
  });

  State._cyModel.once("layoutstop", () => State._cyModel.fit(undefined, 16));
  State._cyModel.on("tap", "node", (evt) => {
    const n = evt.target;
    State._cyModel.animate({ center: { eles: n }, zoom: Math.min(State._cyModel.zoom() * 1.1, 2.5) }, { duration: 180 });
  });
}


// ═══════════════════════════════════════════════════════════════
// CODEGEN VIEW (Kernel Studio)
// ═══════════════════════════════════════════════════════════════

function transpileOp(op, backend, index) {
  const { type, inputs, output } = op;
  const in0 = inputs[0] || "in0", in1 = inputs[1] || "in1";
  const out = output || `t${index !== undefined ? index : ""}`;

  const gpuMap = {
    ADD: `${out} = ${in0} + ${in1};`, SUB: `${out} = ${in0} - ${in1};`,
    MUL: `${out} = ${in0} * ${in1};`, DIV: `${out} = ${in0} / ${in1};`,
    NEG: `${out} = -${in0};`, RECIP: `${out} = 1.0f / ${in0};`,
    EXP: `${out} = exp(${in0});`, LOG: `${out} = log(${in0});`,
    SQRT: `${out} = sqrt(${in0});`, ABS: `${out} = abs(${in0});`,
    SIN: `${out} = sin(${in0});`, COS: `${out} = cos(${in0});`,
    MAX: `${out} = max(${in0}, ${in1});`, MIN: `${out} = min(${in0}, ${in1});`,
    RELU: `${out} = max(${in0}, 0.0f);`,
    SIGMOID: `${out} = 1.0f / (1.0f + exp(-${in0}));`,
    UNKNOWN: `${out} = 1.0f / (1.0f + exp(-${in0}));`,
    TANH: `${out} = tanh(${in0});`,
    MATMUL: `// Matrix Multiplication\n    ${out} = 0.0f;\n    for (int k = 0; k < K; ++k) {\n        ${out} += ${in0}[row * K + k] * ${in1}[k * N + col];\n    }`,
    MEAN: `// Reduction: Mean\n    float sum = 0.0f;\n    for (int j = 0; j < n; j++) sum += ${in0}[j];\n    ${out} = sum / (float)n;`,
    SUM: `// Reduction: Sum\n    float sum = 0.0f;\n    for (int j = 0; j < n; j++) sum += ${in0}[j];\n    ${out} = sum;`,
    FILL: `${out} = 0.0f; // Constant fill`,
  };
  const viewOps = ["PERMUTE", "RESHAPE", "EXPAND", "SLICE", "STRIDE"];
  const cMap = {
    ADD: `${out}[i] = ${in0}[i] + ${in1}[i];`, SUB: `${out}[i] = ${in0}[i] - ${in1}[i];`,
    MUL: `${out}[i] = ${in0}[i] * ${in1}[i];`, DIV: `${out}[i] = ${in0}[i] / ${in1}[i];`,
    NEG: `${out}[i] = -${in0}[i];`, RECIP: `${out}[i] = 1.0f / ${in0}[i];`,
    EXP: `${out}[i] = expf(${in0}[i]);`, LOG: `${out}[i] = logf(${in0}[i]);`,
    SQRT: `${out}[i] = sqrtf(${in0}[i]);`, ABS: `${out}[i] = fabsf(${in0}[i]);`,
    SIN: `${out}[i] = sinf(${in0}[i]);`, COS: `${out}[i] = cosf(${in0}[i]);`,
    MAX: `${out}[i] = fmaxf(${in0}[i], ${in1}[i]);`, MIN: `${out}[i] = fminf(${in0}[i], ${in1}[i]);`,
    RELU: `${out}[i] = fmaxf(${in0}[i], 0.0f);`,
    SIGMOID: `${out}[i] = 1.0f / (1.0f + expf(-${in0}[i]));`,
    UNKNOWN: `${out}[i] = 1.0f / (1.0f + expf(-${in0}[i]));`,
    TANH: `${out}[i] = tanhf(${in0}[i]);`,
    MATMUL: `for (int m = 0; m < M; m++) {\n        for (int n = 0; n < N; n++) {\n            float sum = 0.0f;\n            for (int k = 0; k < K; k++) sum += ${in0}[m*K+k] * ${in1}[k*N+n];\n            ${out}[m*N+n] = sum;\n        }\n    }`,
    MEAN: `float sum = 0.0f;\n        for (int j = 0; j < n; j++) sum += ${in0}[j];\n        ${out}[0] = sum / (float)n;`,
    SUM: `float sum = 0.0f;\n        for (int j = 0; j < n; j++) sum += ${in0}[j];\n        ${out}[0] = sum;`,
    FILL: `${out}[i] = 0.0f; // Constant fill`,
  };
  const simdMap = {
    ADD: `_mm256_store_ps(${out}, _mm256_add_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`,
    SUB: `_mm256_store_ps(${out}, _mm256_sub_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`,
    MUL: `_mm256_store_ps(${out}, _mm256_mul_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`,
    DIV: `_mm256_store_ps(${out}, _mm256_div_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`,
    MAX: `_mm256_store_ps(${out}, _mm256_max_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`,
    MIN: `_mm256_store_ps(${out}, _mm256_min_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`,
    SQRT: `_mm256_store_ps(${out}, _mm256_sqrt_ps(_mm256_load_ps(${in0})));`,
  };

  if (viewOps.includes(type)) {
    const tag = (backend === "c" || backend === "cpu" || backend === "c_simd") ? "View" : "View operation";
    return `// ${tag}: ${type} (${backend === "c" || backend === "cpu" ? "zero-copy" : "memory layout change"})`;
  }

  if (backend === "c_simd") return simdMap[type] || `// ${type}: Scalar fallback needed for SIMD`;
  if (backend === "c" || backend === "cpu") return cMap[type] || `// ${type}: Custom kernel`;
  return gpuMap[type] || `// ${type}: Custom operation`;
}

function generateKernelCode(kernel, backend) {
  const isFused = kernel.isFused || (kernel.ops && kernel.ops.length > 0);
  const name = kernel.name;
  const inputs = kernel.inputs || [];
  const output = kernel.output || "out";
  let code = "";

  if (backend === "cuda") {
    code += `__global__ void ${name}(${[...inputs.map(n => `float* ${n}`), `float* ${output}`, "int n"].join(", ")}) {\n`;
    code += `    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx >= n) return;\n\n`;
  } else if (backend === "metal") {
    code += `kernel void ${name}(\n`;
    inputs.forEach((inp, i) => { code += `    device const float* ${inp} [[buffer(${i})]],\n`; });
    code += `    device float* ${output} [[buffer(${inputs.length})]],\n    uint id [[thread_position_in_grid]]) {\n    if (id >= n) return;\n\n`;
  } else if (backend === "opencl") {
    code += `__kernel void ${name}(${[...inputs.map(n => `__global const float* ${n}`), `__global float* ${output}`, "int n"].join(", ")}) {\n`;
    code += `    int idx = get_global_id(0);\n    if (idx >= n) return;\n\n`;
  } else if (backend === "wgsl") {
    code += `@compute @workgroup_size(64)\nfn ${name}(@builtin(global_invocation_id) id: vec3<u32>) {\n`;
    code += `    let idx = id.x;\n    if (idx >= n) { return; }\n\n`;
  } else {
    code += `void ${name}(float** inputs_ptr, float** outputs_ptr, int n) {\n`;
    inputs.forEach((inp, i) => { code += `    float* ${inp} = inputs_ptr[${i}];\n`; });
    code += `    float* ${output} = outputs_ptr[0];\n\n`;
    if (backend === "cpu") code += `    #pragma omp parallel for\n`;
    code += `    for (int i = 0; i < n; i++) {\n`;
  }

  if (isFused && kernel.ops) {
    code += `        // Fused Kernel Body\n`;
    kernel.ops.forEach((op, idx) => {
      const indexer = (backend === "c" || backend === "cpu") ? "[i]" : (backend === "metal" ? "[id]" : "[idx]");
      const mappedInputs = op.inputs.map(inp => inputs.includes(inp) ? `${inp}${indexer}` : inp);
      const mappedOutput = (op.output === output) ? `${op.output}${indexer}` : op.output;
      const mappedOp = { ...op, inputs: mappedInputs, output: mappedOutput };
      const indent = (backend === "c" || backend === "cpu") ? "        " : "    ";
      let line = transpileOp(mappedOp, backend, idx);
      if (op.output !== output && line.match(/^\w+\s*=/)) line = `float ${line}`;
      code += `${indent}${line}\n`;
    });
  } else {
    const indexer = (backend === "c" || backend === "cpu") ? "[i]" : (backend === "metal" ? "[id]" : "[idx]");
    const mappedInputs = inputs.map(inp => `${inp}${indexer}`);
    const mappedOutput = `${output}${indexer}`;
    const indent = (backend === "c" || backend === "cpu") ? "        " : "    ";
    code += `${indent}${transpileOp({ type: kernel.type, inputs: mappedInputs, output: mappedOutput }, backend)}\n`;
  }

  if (backend === "c" || backend === "cpu") code += `    }\n`;
  code += `}\n`;
  return code;
}

function renderCodeGenView() {
  const container = $("#tab-codegen");
  const data = State.kernels;
  const hasKernelData = data &&
    ((data.unoptimized?.kernels?.length > 0) || (data.optimized?.kernels?.length > 0));

  container.innerHTML = "";

  if (!hasKernelData) {
    renderCodeGenEmpty(container);
    return;
  }

  const kernelData = data;
  const acc = State.activeAccelerator;

  // Layout
  const layout = el("div", { className: "codegen-layout" });

  // Sidebar
  const sidebar = el("div", { className: "codegen-sidebar" });
  sidebar.appendChild(el("div", { className: "codegen-sidebar-header" }, el("h2", {}, "Target Accelerators")));

  const accelerators = [
    { id: "c", name: "C (Scalar)", icon: '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>', desc: "Portable C99 implementation" },
    { id: "c_simd", name: "C (SIMD)", icon: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>', desc: "AVX2/NEON vector intrinsics" },
    { id: "cuda", name: "NVIDIA CUDA", icon: '<rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/>', desc: "High-performance GPU backend" },
    { id: "metal", name: "Apple Metal", icon: '<rect x="5" y="2" width="14" height="20" rx="2" ry="2"/><line x1="12" y1="18" x2="12.01" y2="18"/>', desc: "Optimized for Apple Silicon" },
    { id: "opencl", name: "OpenCL", icon: '<rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>', desc: "Cross-platform GPU acceleration" },
    { id: "wgsl", name: "WebGPU (WGSL)", icon: '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>', desc: "Next-gen web graphics" },
    { id: "cpu", name: "OpenMP CPU", icon: '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>', desc: "Multi-threaded CPU fallback" },
  ];

  const list = el("div", { className: "codegen-sidebar-list" });
  accelerators.forEach(a => {
    const btn = el("button", { className: "acc-btn" + (a.id === acc ? " active" : "") });
    btn.appendChild(el("div", { className: "acc-btn-icon", innerHTML: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${a.icon}</svg>` }));
    const info = el("div", {});
    info.appendChild(el("div", { className: "acc-btn-name" }, a.name));
    info.appendChild(el("div", { className: "acc-btn-desc" }, a.desc));
    btn.appendChild(info);
    btn.addEventListener("click", () => {
      State.activeAccelerator = a.id;
      renderCodeGenView();
    });
    list.appendChild(btn);
  });
  sidebar.appendChild(list);

  // Stats
  const stats = el("div", { className: "codegen-stats" });
  stats.appendChild(el("div", { className: "codegen-stats-label" }, "OPTIMIZATION STATS"));
  const deadRow = el("div", { className: "codegen-stat-row" });
  deadRow.appendChild(el("span", {}, "Dead Code"));
  deadRow.appendChild(el("span", { style: { color: "var(--accent-error)" } }, `${kernelData.unoptimized?.deadNodes || 0} nodes`));
  stats.appendChild(deadRow);
  const fusedRow = el("div", { className: "codegen-stat-row" });
  fusedRow.appendChild(el("span", {}, "Fused Kernels"));
  fusedRow.appendChild(el("span", { style: { color: "var(--accent-success)" } }, `${kernelData.optimized?.fusedKernels || 0} kernels`));
  stats.appendChild(fusedRow);
  sidebar.appendChild(stats);

  layout.appendChild(sidebar);

  // Main content
  const main = el("div", { className: "codegen-main" });

  // Toolbar
  const ext = acc === "cuda" ? "cu" : acc === "metal" ? "metal" : acc === "wgsl" ? "wgsl" : acc === "opencl" ? "cl" : "c";
  const toolbar = el("div", { className: "codegen-toolbar" });
  const fnDiv = el("div", { className: "codegen-toolbar-filename" });
  fnDiv.appendChild(svgIcon('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>'));
  fnDiv.firstChild.style.color = "var(--accent-primary)";
  fnDiv.appendChild(el("span", {}, `generated_module_${acc}.${ext}`));
  toolbar.appendChild(fnDiv);
  main.appendChild(toolbar);

  // Code panes
  const panes = el("div", { className: "codegen-panes" });

  function makeCodePane(title, badge, badgeColor, source) {
    const pane = el("div", { className: "code-pane" });
    const header = el("div", { className: "code-pane-header" });
    header.appendChild(el("span", {}, title));
    header.appendChild(el("span", { style: { color: badgeColor, fontSize: "10px" } }, badge));
    pane.appendChild(header);

    const body = el("div", { className: "code-pane-body" });
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    // highlight.js
    const highlighted = hljs.highlight(source, { language: "c" }).value;
    // Wrap each line for line numbers
    const lines = highlighted.split("\n");
    code.innerHTML = lines.map(l => `<span class="code-line">${l}</span>`).join("\n");
    pre.appendChild(code);
    body.appendChild(pre);
    pane.appendChild(body);
    return pane;
  }

  const originalSource = (kernelData.unoptimized?.kernels?.length)
    ? kernelData.unoptimized.kernels.map(k => {
        let h = `// Kernel: ${k.name} (${k.type})`;
        if (k.isDead) h += " [DEAD CODE - REMOVED]";
        return `${h}\n${generateKernelCode(k, acc)}`;
      }).join("\n\n")
    : "// No original kernels available";

  const optimizedSource = (kernelData.optimized?.kernels?.length)
    ? kernelData.optimized.kernels.map(k => {
        let h = `// Kernel: ${k.name} (${k.type})`;
        if (k.isFused) h += " [FUSED KERNEL]";
        return `${h}\n${generateKernelCode(k, acc)}`;
      }).join("\n\n")
    : "// No optimized kernels available";

  panes.appendChild(makeCodePane(
    "ORIGINAL IR (Unoptimized)",
    `${kernelData.unoptimized?.deadNodes || 0} DEAD NODES`,
    "var(--accent-error)",
    originalSource
  ));
  panes.appendChild(makeCodePane(
    "OPTIMIZED KERNELS (Fused)",
    `${kernelData.optimized?.fusedKernels || 0} FUSED KERNELS`,
    "var(--accent-success)",
    optimizedSource
  ));

  main.appendChild(panes);
  layout.appendChild(main);
  container.appendChild(layout);
}

function renderCodeGenEmpty(container) {
  const wrap = el("div", { style: { height: "100%", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg-dark)" } });
  const box = el("div", { className: "empty-state-box" });

  box.appendChild(el("div", { className: "empty-state-icon",
    innerHTML: '<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>'
  }));

  const text = el("div", { style: { textAlign: "center" } });
  text.appendChild(el("div", { className: "empty-state-title" }, "No Kernel Data"));
  text.appendChild(el("div", { className: "empty-state-desc" }, "Kernel Studio generates optimized code for multiple accelerator backends."));
  box.appendChild(text);

  const grid = el("div", { className: "backend-grid" });
  [
    { icon: '<rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/>', label: "CUDA" },
    { icon: '<rect x="5" y="2" width="14" height="20" rx="2"/><line x1="12" y1="18" x2="12.01" y2="18"/>', label: "Metal" },
    { icon: '<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>', label: "OpenCL" },
    { icon: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>', label: "CPU" },
    { icon: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>', label: "SIMD" },
    { icon: '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>', label: "WebGPU" },
  ].forEach(item => {
    const gi = el("div", { className: "backend-grid-item" });
    gi.appendChild(svgIcon(item.icon, 14, 14));
    gi.appendChild(el("span", {}, item.label));
    grid.appendChild(gi);
  });
  box.appendChild(grid);

  const features = el("div", { className: "feature-list" });
  [
    { icon: '<polyline points="3 6 5 6 6 12"/><line x1="10" y1="12.76" x2="10" y2="12.76"/><circle cx="8.5" cy="18.5" r="1.5"/><circle cx="16.5" cy="18.5" r="1.5"/><path d="M6 12h13l-1.5-8H7"/>', color: "#ef4444", text: "Dead code elimination" },
    { icon: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>', color: "#10b981", text: "Kernel fusion optimization" },
    { icon: '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>', color: "#f59e0b", text: "Side-by-side comparison" },
  ].forEach(item => {
    const fi = el("div", { className: "feature-list-item" });
    const iconSpan = svgIcon(item.icon, 14, 14);
    iconSpan.style.color = item.color;
    fi.appendChild(iconSpan);
    fi.appendChild(el("span", {}, item.text));
    features.appendChild(fi);
  });
  box.appendChild(features);

  const hint = el("div", { className: "hint-box" });
  hint.appendChild(svgIcon('<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>', 16, 16));
  hint.firstChild.style.color = "#10b981";
  hint.appendChild(el("code", {}, "VIZ=1 ./build/bin/dead_code_example"));
  box.appendChild(hint);

  wrap.appendChild(box);
  container.appendChild(wrap);
}


// ═══════════════════════════════════════════════════════════════
// TRAINING VIEW
// ═══════════════════════════════════════════════════════════════

function renderTrainingView() {
  const container = $("#tab-training");
  const data = State.training;

  container.innerHTML = "";

  if (!data || data.error) {
    renderTrainingEmpty(container, data);
    return;
  }

  const layout = el("div", { className: "training-layout" });

  // ── Extract data ──────────────────────────────────────────
  const d = data;
  const trainingLosses = d.epoch_training_losses || d.epoch_train_losses || d.epoch_losses || [];
  const testingLosses = d.epoch_testing_losses || d.epoch_test_losses || d.epoch_losses || [];
  const trainingAccuracies = d.epoch_training_accuracies || d.epoch_train_accuracies || d.epoch_accuracies || [];
  const testingAccuracies = d.epoch_testing_accuracies || d.epoch_test_accuracies || d.epoch_accuracies || [];

  const completedTrainingLosses = trainingLosses.filter(l => l != null && l >= 0);
  const completedTestingLosses = testingLosses.filter(l => l != null && l >= 0);
  const completedTrainingAccuracies = trainingAccuracies.filter(a => a != null && a >= 0);
  const completedTestingAccuracies = testingAccuracies.filter(a => a != null && a >= 0);
  const completedEpochs = Math.max(completedTrainingLosses.length, completedTestingLosses.length, completedTrainingAccuracies.length, completedTestingAccuracies.length);

  const validationLosses = d.epoch_validation_losses || d.epoch_val_losses || [];
  const completedValidationLosses = validationLosses.filter(l => l != null && l >= 0);
  const validationAccuracies = d.epoch_validation_accuracies || d.epoch_val_accuracies || [];
  const completedValidationAccuracies = validationAccuracies.filter(a => a != null && a >= 0);

  const hasTestingData = !!(
    (d.epoch_testing_losses && d.epoch_testing_losses.some(v => v > 0)) ||
    (d.epoch_test_losses && d.epoch_test_losses.some(v => v > 0)) ||
    (d.epoch_testing_accuracies && d.epoch_testing_accuracies.some(v => v > 0)) ||
    (d.epoch_test_accuracies && d.epoch_test_accuracies.some(v => v > 0)) ||
    (d.testing_loss && d.testing_loss > 0) || (d.test_loss && d.test_loss > 0) ||
    (d.testing_accuracy && d.testing_accuracy > 0) || (d.test_accuracy && d.test_accuracy > 0)
  );

  const hasValidationData = !!(d.validation_loss || d.val_loss ||
    (d.epoch_validation_losses && d.epoch_validation_losses.length > 0 && d.epoch_validation_losses.some(v => v > 0)) ||
    (d.epoch_val_losses && d.epoch_val_losses.length > 0 && d.epoch_val_losses.some(v => v > 0)) ||
    d.validation_accuracy || d.val_accuracy ||
    (d.epoch_validation_accuracies && d.epoch_validation_accuracies.length > 0 && d.epoch_validation_accuracies.some(v => v > 0)) ||
    (d.epoch_val_accuracies && d.epoch_val_accuracies.length > 0 && d.epoch_val_accuracies.some(v => v > 0)) ||
    completedValidationLosses.length > 0 || completedValidationAccuracies.length > 0);

  const showCheckboxes = hasTestingData || hasValidationData;

  // Latest metrics
  const latestTrainingLoss = d.training_loss ?? d.train_loss ?? (completedTrainingLosses.length > 0 ? completedTrainingLosses[completedTrainingLosses.length - 1] : 0);
  const latestTestingLoss = (() => {
    if (d.testing_loss) return d.testing_loss;
    if (d.test_loss) return d.test_loss;
    for (let i = completedTestingLosses.length - 1; i >= 0; i--) { if (completedTestingLosses[i] > 0) return completedTestingLosses[i]; }
    return null;
  })();
  const latestValidationLoss = d.validation_loss ?? d.val_loss ?? (completedValidationLosses.length > 0 ? completedValidationLosses[completedValidationLosses.length - 1] : null);
  const latestTrainingAccuracy = d.training_accuracy ?? d.train_accuracy ?? (completedTrainingAccuracies.length > 0 ? completedTrainingAccuracies[completedTrainingAccuracies.length - 1] : 0);
  const latestTestingAccuracy = (() => {
    if (d.testing_accuracy) return d.testing_accuracy;
    if (d.test_accuracy) return d.test_accuracy;
    for (let i = completedTestingAccuracies.length - 1; i >= 0; i--) { if (completedTestingAccuracies[i] > 0) return completedTestingAccuracies[i]; }
    return null;
  })();
  const latestValidationAccuracy = d.validation_accuracy ?? d.val_accuracy ?? (completedValidationAccuracies.length > 0 ? completedValidationAccuracies[completedValidationAccuracies.length - 1] : null);

  // Chart data
  const learningRate = d.learning_rate ?? d.lr ?? d.current_lr ?? null;
  const lrHistory = d.epoch_learning_rates || d.lr_history || d.learning_rate_history || null;
  const effectiveMaxEpochs = (d.early_stopped && d.actual_epochs)
    ? d.actual_epochs
    : Math.max(completedEpochs, completedValidationLosses.length, completedValidationAccuracies.length, completedTestingLosses.length, completedTestingAccuracies.length, d.num_epochs || completedEpochs);

  const chartData = Array.from({ length: effectiveMaxEpochs }, (_, i) => {
    let testLoss = null, testAcc = null;
    if (i < completedTestingLosses.length && completedTestingLosses[i] > 0) testLoss = completedTestingLosses[i];
    else if (i === effectiveMaxEpochs - 1 && (d.testing_loss || d.test_loss)) testLoss = d.testing_loss || d.test_loss;
    if (i < completedTestingAccuracies.length && completedTestingAccuracies[i] > 0) testAcc = completedTestingAccuracies[i];
    else if (i === effectiveMaxEpochs - 1 && (d.testing_accuracy || d.test_accuracy)) testAcc = d.testing_accuracy || d.test_accuracy;

    return {
      epoch: i + 1,
      trainingLoss: completedTrainingLosses[i] != null ? completedTrainingLosses[i] : null,
      testingLoss: testLoss,
      validationLoss: completedValidationLosses[i] != null ? completedValidationLosses[i] : null,
      trainingAccuracy: completedTrainingAccuracies[i] != null ? completedTrainingAccuracies[i] * 100 : null,
      testingAccuracy: testAcc != null ? testAcc * 100 : null,
      validationAccuracy: completedValidationAccuracies[i] != null ? completedValidationAccuracies[i] * 100 : null,
    };
  });

  // ── Header ─────────────────────────────────────────────────
  const header = el("div", { className: "training-header" });
  header.appendChild(el("h3", {}, "Training Results"));

  const statusWrap = el("div", { style: { display: "flex", gap: "8px", alignItems: "center" } });
  const effectiveNumEpochs = d.num_epochs || completedEpochs;
  const effectiveActualEpochs = d.actual_epochs || completedEpochs;
  const effectiveExpectedEpochs = d.expected_epochs || effectiveNumEpochs;

  let statusText, statusClass;
  if (d.is_training) { statusText = `Training... (${d.current_epoch || completedEpochs}/${effectiveNumEpochs})`; statusClass = "training"; }
  else if (d.early_stopped) { statusText = `Early Stopped (${effectiveActualEpochs}/${effectiveExpectedEpochs})`; statusClass = "early-stopped"; }
  else { statusText = `Completed (${d.current_epoch || completedEpochs}/${effectiveNumEpochs})`; statusClass = "completed"; }

  statusWrap.appendChild(el("div", { className: `status-badge ${statusClass}` }, statusText));
  if (d.early_stopped) {
    const tag = el("div", { className: "early-stop-tag" });
    tag.appendChild(svgIcon('<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>', 14, 14));
    tag.appendChild(el("span", {}, "Early Stop"));
    statusWrap.appendChild(tag);
  }
  header.appendChild(statusWrap);
  layout.appendChild(header);

  // ── Metric Cards ───────────────────────────────────────────
  const cardCount = 2 + (hasTestingData ? 2 : 0) + (hasValidationData ? 2 : 0);
  const cards = el("div", { className: "metric-cards", style: { gridTemplateColumns: `repeat(${cardCount}, 1fr)` } });

  function addCard(label, value, colorClass) {
    const c = el("div", { className: `metric-card ${colorClass}` });
    c.appendChild(el("div", { className: "metric-card-label" }, label));
    c.appendChild(el("div", { className: "metric-card-value" }, value));
    cards.appendChild(c);
  }

  addCard("Training Loss", latestTrainingLoss?.toFixed(6) || "N/A", "indigo");
  if (hasTestingData) addCard("Testing Loss", latestTestingLoss != null ? latestTestingLoss.toFixed(6) : "N/A", "amber");
  if (hasValidationData) addCard("Validation Loss", latestValidationLoss?.toFixed(6) || "N/A", "red");
  addCard("Training Accuracy", (latestTrainingAccuracy * 100)?.toFixed(2) + "%" || "N/A", "emerald");
  if (hasTestingData) addCard("Testing Accuracy", latestTestingAccuracy != null ? (latestTestingAccuracy * 100).toFixed(2) + "%" : "N/A", "amber");
  if (hasValidationData) addCard("Validation Accuracy", (latestValidationAccuracy * 100)?.toFixed(2) + "%" || "N/A", "red");
  layout.appendChild(cards);

  // ── Charts ─────────────────────────────────────────────────
  const chartsRow = el("div", { className: "charts-row" });

  function makeChartCard(type, title, colorClass) {
    const card = el("div", { className: `chart-card ${colorClass}` });
    const hdr = el("div", { className: "chart-header" });
    const h4 = el("h4", {}, title);
    if (d.early_stopped) {
      const esTag = el("div", { style: { display: "inline-flex", alignItems: "center", gap: "4px", padding: "2px 6px", borderRadius: "3px", fontSize: "9px", fontWeight: "600", background: "rgba(245,158,11,0.15)", color: "#f59e0b", border: "1px solid rgba(245,158,11,0.3)" } });
      esTag.appendChild(svgIcon('<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>', 10, 10));
      esTag.appendChild(el("span", {}, "Early Stop"));
      h4.appendChild(esTag);
    }
    hdr.appendChild(h4);

    if (showCheckboxes) {
      const toggles = el("div", { className: "chart-toggles" });
      const isAcc = type === "accuracy";
      const trainKey = isAcc ? "showTrainingAcc" : "showTrainingLoss";
      const testKey = isAcc ? "showTestingAcc" : "showTestingLoss";
      const valKey = isAcc ? "showValidationAcc" : "showValidationLoss";
      const trainColor = isAcc ? "#10b981" : "#4a90e2";
      const testColor = "#f59e0b";
      const valColor = "#ef4444";

      function addToggle(label, stateKey, color, show) {
        if (!show) return;
        const lbl = el("label", { className: "chart-toggle" });
        const cb = el("input", { type: "checkbox" });
        cb.checked = State[stateKey];
        cb.style.accentColor = color;
        cb.addEventListener("change", () => {
          State[stateKey] = cb.checked;
          renderCharts(chartData, hasTestingData, hasValidationData, showCheckboxes, d);
        });
        lbl.appendChild(cb);
        const span = el("span", { style: { display: "flex", alignItems: "center", gap: "4px" } });
        span.appendChild(el("span", { className: "chart-toggle-line", style: { background: color } }));
        span.appendChild(el("span", {}, label));
        lbl.appendChild(span);
        toggles.appendChild(lbl);
      }

      addToggle("Training", trainKey, trainColor, true);
      addToggle("Testing", testKey, testColor, hasTestingData);
      addToggle("Validation", valKey, valColor, hasValidationData);
      hdr.appendChild(toggles);
    }

    card.appendChild(hdr);
    const area = el("div", { className: "chart-area", id: `chart-${type}` });
    card.appendChild(area);
    return card;
  }

  chartsRow.appendChild(makeChartCard("loss", "Loss Curve", "loss"));
  chartsRow.appendChild(makeChartCard("accuracy", "Accuracy Curve", "accuracy"));
  layout.appendChild(chartsRow);

  // ── Bottom row: Metrics + Epoch Table ──────────────────────
  const bottomRow = el("div", { className: "bottom-row" });

  // Metrics Panel
  const metricsPanel = el("div", { className: "metrics-panel" });
  const mpTitle = el("h4", { className: "metrics-panel-title" });
  const mpLeft = el("div", { style: { display: "flex", alignItems: "center", gap: "8px" } });
  mpLeft.appendChild(el("span", { className: "accent-bar" }));
  mpLeft.appendChild(document.createTextNode("Training Metrics"));
  mpTitle.appendChild(mpLeft);

  // Convergence badge
  const lossTrend = (() => {
    if (completedTrainingLosses.length < 3) return "insufficient";
    const recent = completedTrainingLosses.slice(-5);
    const older = completedTrainingLosses.slice(-10, -5);
    if (older.length === 0) return "insufficient";
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
    const change = ((olderAvg - recentAvg) / olderAvg) * 100;
    if (change > 1) return "improving";
    if (change < -1) return "degrading";
    return "plateau";
  })();
  const trendColors = { improving: "#10b981", degrading: "#ef4444", plateau: "#f59e0b", insufficient: "#6b7280" };
  const trendLabels = { improving: "Converging", degrading: "Diverging", plateau: "Plateau", insufficient: "Init" };
  const trendArrows = { improving: "\u2193", degrading: "\u2191", plateau: "\u2192", insufficient: "\u2014" };

  const badge = el("div", { className: "convergence-badge", style: { background: `${trendColors[lossTrend]}20`, border: `1px solid ${trendColors[lossTrend]}60` } });
  badge.appendChild(el("div", { className: "convergence-dot", style: { background: trendColors[lossTrend], boxShadow: `0 0 6px ${trendColors[lossTrend]}80` } }));
  badge.appendChild(el("span", { style: { color: "var(--text)", fontWeight: "600" } }, trendArrows[lossTrend]));
  badge.appendChild(el("span", { style: { color: "var(--text)", fontWeight: "500", marginLeft: "2px" } }, trendLabels[lossTrend]));
  mpTitle.appendChild(badge);
  metricsPanel.appendChild(mpTitle);

  const scroll = el("div", { className: "metrics-scroll" });

  // Progress bar
  const currentEpochNum = d.current_epoch || completedEpochs || 0;
  const totalEpochs = d.early_stopped ? effectiveActualEpochs : (d.num_epochs || 1);
  const epochProgress = totalEpochs > 0 ? (currentEpochNum / totalEpochs) * 100 : 0;

  const progressSection = el("div", { className: "progress-section" });
  const progressLabel = el("div", { className: "progress-label" });
  progressLabel.appendChild(el("span", {}, "Epoch Progress"));
  progressLabel.appendChild(el("span", {}, `${currentEpochNum}/${totalEpochs}`));
  progressSection.appendChild(progressLabel);
  const barBg = el("div", { className: "progress-bar-bg" });
  barBg.appendChild(el("div", { className: "progress-bar-fill", style: { width: `${epochProgress}%` } }));
  progressSection.appendChild(barBg);
  scroll.appendChild(progressSection);

  // Metrics grid
  const epochTime = d.epoch_time ?? d.time_per_epoch ?? null;
  const totalTime = d.total_time ?? d.elapsed_time ?? null;
  const avgEpochTime = (epochTime && epochTime > 0) ? epochTime : (totalTime && totalTime > 0 && completedEpochs > 0) ? totalTime / completedEpochs : null;
  const estimatedRemaining = (d.estimated_remaining && d.estimated_remaining > 0) ? d.estimated_remaining
    : (avgEpochTime && totalEpochs > currentEpochNum ? avgEpochTime * (totalEpochs - currentEpochNum) : null);
  const epochsPerHour = (d.epochs_per_hour && d.epochs_per_hour > 0) ? d.epochs_per_hour : (avgEpochTime && avgEpochTime > 0 ? 3600 / avgEpochTime : null);
  const lr = d.learning_rate ?? d.lr ?? d.current_lr ?? null;
  const lrSchedule = d.lr_schedule || d.learning_rate_schedule || d.scheduler || null;
  const gradientNorm = d.gradient_norm ?? d.grad_norm ?? d.gradient_norm_avg ?? null;
  const lossReductionRate = (() => {
    if (d.loss_reduction_rate != null) return d.loss_reduction_rate;
    if (completedTrainingLosses.length < 2) return null;
    const last = completedTrainingLosses[completedTrainingLosses.length - 1];
    const prev = completedTrainingLosses[completedTrainingLosses.length - 2];
    if (prev === 0) return null;
    return ((prev - last) / prev) * 100;
  })();
  const lossStability = (() => {
    if (d.loss_stability != null) return d.loss_stability;
    if (completedTrainingLosses.length < 5) return null;
    const recent = completedTrainingLosses.slice(-10);
    const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
    const variance = recent.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / recent.length;
    return Math.sqrt(variance);
  })();

  function formatTime(s) {
    if (s >= 3600) return `${(s / 3600).toFixed(2)}hr`;
    if (s >= 60) return `${(s / 60).toFixed(2)}min`;
    return `${s.toFixed(2)}s`;
  }

  function formatEpochsPerHour(eph) {
    if (eph >= 1) return `${eph.toFixed(1)}/hr`;
    if (eph * 60 >= 1) return `${(eph * 60).toFixed(1)}/min`;
    return `${(eph * 3600).toFixed(1)}/sec`;
  }

  const gradientHealth = (() => {
    if (!gradientNorm) return null;
    if (gradientNorm > 100) return { color: "#ef4444", text: "Exploding" };
    if (gradientNorm < 0.001) return { color: "#ef4444", text: "Vanishing" };
    if (gradientNorm < 0.01) return { color: "#f59e0b", text: "Low" };
    return { color: "#10b981", text: "Healthy" };
  })();

  const grid = el("div", { className: "metrics-grid" });

  function addMetric(label, value) {
    const item = el("div", {});
    item.appendChild(el("div", { className: "metric-item-label" }, label));
    if (typeof value === "string") {
      item.appendChild(el("div", { className: "metric-item-value" }, value));
    } else {
      item.appendChild(value);
    }
    grid.appendChild(item);
  }

  addMetric("Time/Epoch", avgEpochTime ? formatTime(avgEpochTime) : "N/A");
  addMetric("Total Time", totalTime ? `${(totalTime / 60).toFixed(1)}m` : "N/A");
  addMetric("Est. Remaining", estimatedRemaining ? `${(estimatedRemaining / 60).toFixed(1)}m` : "N/A");
  addMetric("Epochs/Hour", epochsPerHour ? formatEpochsPerHour(epochsPerHour) : "N/A");

  // Learning rate with schedule badge
  const lrDiv = el("div", { className: "metric-item-value", style: { display: "flex", alignItems: "center", gap: "4px", flexWrap: "wrap" } });
  if (lr) {
    lrDiv.appendChild(el("span", {}, lr < 0.001 ? lr.toExponential(2) : lr.toFixed(6)));
    const schedLabel = lrSchedule || "Constant";
    const schedBg = lrSchedule ? "rgba(99,102,241,0.15)" : "rgba(107,114,128,0.15)";
    const schedBorder = lrSchedule ? "rgba(99,102,241,0.2)" : "rgba(107,114,128,0.2)";
    lrDiv.appendChild(el("span", { style: { fontSize: "9px", color: "var(--muted)", marginLeft: "4px", padding: "2px 6px", background: schedBg, borderRadius: "4px", border: `1px solid ${schedBorder}` } }, schedLabel));
  } else {
    lrDiv.textContent = "N/A";
  }
  addMetric("Learning Rate", lrDiv);

  // Gradient Health
  const ghDiv = el("div", { className: "metric-item-value", style: { display: "flex", alignItems: "center", gap: "6px" } });
  if (gradientHealth) {
    ghDiv.appendChild(el("span", { style: { width: "8px", height: "8px", borderRadius: "50%", background: gradientHealth.color, boxShadow: `0 0 6px ${gradientHealth.color}80` } }));
    ghDiv.appendChild(el("span", { style: { color: gradientHealth.color, fontWeight: "600" } }, gradientHealth.text));
    if (gradientNorm) ghDiv.appendChild(el("span", { style: { fontSize: "10px", color: "var(--muted)", fontFamily: "monospace", marginLeft: "4px" } }, `(${gradientNorm.toFixed(4)})`));
  } else {
    ghDiv.textContent = "N/A";
  }
  addMetric("Gradient Health", ghDiv);

  addMetric("Reduction Rate", lossReductionRate !== null ? `${lossReductionRate.toFixed(2)}%` : "N/A");
  addMetric("Loss Stability (\u03c3)", lossStability !== null ? lossStability.toFixed(6) : "N/A");

  scroll.appendChild(grid);

  // Throughput
  const throughput = d.throughput || d.samples_per_sec || d.tokens_per_sec || null;
  if (throughput) {
    const tpDiv = el("div", { style: { marginTop: "8px", paddingTop: "8px", borderTop: "1px solid rgba(16,185,129,0.1)" } });
    tpDiv.appendChild(el("div", { className: "metric-item-label" }, "Throughput"));
    tpDiv.appendChild(el("div", { className: "metric-item-value" }, `${throughput.toLocaleString()} ${d.tokens_per_sec ? "tokens/s" : "samples/s"}`));
    scroll.appendChild(tpDiv);
  }

  metricsPanel.appendChild(scroll);
  bottomRow.appendChild(metricsPanel);

  // ── Epoch Table ────────────────────────────────────────────
  const tableCard = el("div", { className: "epoch-table-card" });
  const ttl = el("h4", { className: "epoch-table-title" });
  ttl.appendChild(el("span", { className: "accent-bar" }));
  ttl.appendChild(document.createTextNode("Epoch Summary"));
  tableCard.appendChild(ttl);

  const tableWrap = el("div", { className: "epoch-table-wrap" });
  const table = el("table", { className: "epoch-table" });
  const thead = el("thead", {});
  const headRow = el("tr", {});
  headRow.appendChild(el("th", {}, "Epoch"));
  headRow.appendChild(el("th", {}, "Loss"));
  headRow.appendChild(el("th", {}, "Accuracy"));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el("tbody", {});
  completedTrainingLosses.forEach((loss, i) => {
    const isLatest = i === completedTrainingLosses.length - 1;
    const tr = el("tr", { className: isLatest ? "latest" : "" });
    const epochTd = el("td", {}, String(i + 1));
    if (isLatest) epochTd.appendChild(el("span", { className: "epoch-dot" }, "\u25cf"));
    tr.appendChild(epochTd);
    tr.appendChild(el("td", {}, loss.toFixed(6)));
    const acc = completedTrainingAccuracies[i];
    tr.appendChild(el("td", {}, acc != null ? (acc * 100).toFixed(2) + "%" : "N/A"));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  tableWrap.appendChild(table);
  tableCard.appendChild(tableWrap);
  bottomRow.appendChild(tableCard);

  layout.appendChild(bottomRow);
  container.appendChild(layout);

  // Render charts after DOM is ready
  requestAnimationFrame(() => renderCharts(chartData, hasTestingData, hasValidationData, showCheckboxes, d));
}

function renderCharts(chartData, hasTestingData, hasValidationData, showCheckboxes, d) {
  // Clean up old observers
  State._chartObservers.forEach(ro => ro.disconnect());
  State._chartObservers = [];

  const lossArea = document.getElementById("chart-loss");
  const accArea = document.getElementById("chart-accuracy");
  if (!lossArea || !accArea) return;

  function renderLoss() {
    VizCharts.createChart(lossArea, chartData, {
      type: "loss",
      visible: {
        training: State.showTrainingLoss,
        testing: State.showTestingLoss && hasTestingData,
        validation: State.showValidationLoss && hasValidationData,
      }
    });
  }

  function renderAcc() {
    VizCharts.createChart(accArea, chartData, {
      type: "accuracy",
      visible: {
        training: State.showTrainingAcc,
        testing: State.showTestingAcc && hasTestingData,
        validation: State.showValidationAcc && hasValidationData,
      }
    });
  }

  renderLoss();
  renderAcc();

  State._chartObservers.push(VizCharts.observeResize(lossArea, renderLoss));
  State._chartObservers.push(VizCharts.observeResize(accArea, renderAcc));
}

function renderTrainingEmpty(container, data) {
  const wrap = el("div", { style: { padding: "24px", height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "var(--bg-dark)" } });

  if (data && data.error) {
    const errBox = el("div", { className: "error-state" });
    errBox.appendChild(el("div", { className: "error-icon",
      innerHTML: '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
    }));
    errBox.appendChild(el("div", { style: { color: "#ef4444", fontSize: "16px", fontWeight: "600" } }, "Error Loading Data"));
    errBox.appendChild(el("div", { style: { color: "var(--text-secondary)", fontSize: "13px", textAlign: "center" } }, data.error));
    wrap.appendChild(errBox);
  } else {
    const box = el("div", { className: "empty-state-box", style: { maxWidth: "450px" } });

    box.appendChild(el("div", { className: "empty-state-icon",
      innerHTML: '<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
    }));

    const text = el("div", { style: { textAlign: "center" } });
    text.appendChild(el("div", { className: "empty-state-title" }, "No Training Data"));
    text.appendChild(el("div", { className: "empty-state-desc" }, "Training metrics will appear here when you run a training example."));
    box.appendChild(text);

    const features = el("div", { className: "training-features" });
    [
      { label: "Loss Curves", color: "#6366f1" },
      { label: "Accuracy Charts", color: "#10b981" },
      { label: "Time Metrics", color: "#f59e0b" },
      { label: "Convergence", color: "#8b5cf6" },
    ].forEach(item => {
      const fi = el("div", { className: "training-feature-item" });
      fi.appendChild(el("span", { className: "dot", style: { background: item.color } }));
      fi.appendChild(el("span", {}, item.label));
      features.appendChild(fi);
    });
    box.appendChild(features);

    const hint = el("div", { className: "hint-box" });
    hint.appendChild(svgIcon('<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>', 16, 16));
    hint.firstChild.style.color = "#10b981";
    hint.appendChild(el("code", {}, "VIZ=1 ./build/bin/training_loop_example"));
    box.appendChild(hint);

    wrap.appendChild(box);
  }

  container.appendChild(wrap);
}


// ═══════════════════════════════════════════════════════════════
// WORKER (inline — for graph layout offloading)
// ═══════════════════════════════════════════════════════════════
// worker.js is optional; layout runs fine in main thread for typical graph sizes.
// If needed, load viz/worker.js as a Web Worker and postMessage layout data.


// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  // Register cytoscape extensions
  if (typeof cytoscapeDagre !== "undefined") cytoscape.use(cytoscapeDagre);
  if (typeof cytoscapeElk !== "undefined") cytoscape.use(cytoscapeElk);

  // Tab click handlers
  $$(".topbar-button").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // Initial render
  renderGraphView();

  // Connect data for initial tab
  connectDataForTab("graph");
});
