/* worker.js — Web Worker for heavy graph layout computation */
/* Optional: offloads dagre/ELK layout to a background thread */

"use strict";

self.onmessage = function(e) {
  const { type, elements, options } = e.data;

  if (type === "layout") {
    // For dagre layout, we compute node positions here
    // and send them back to the main thread.
    // This is useful for graphs with 100+ nodes.
    try {
      // Import dagre if available in worker scope
      if (typeof dagre !== "undefined") {
        const g = new dagre.graphlib.Graph({ compound: true });
        g.setGraph({
          rankdir: options.rankDir || "LR",
          nodesep: options.nodeSep || 50,
          ranksep: options.rankSep || 70,
          edgesep: options.edgeSep || 30,
          ranker: options.ranker || "network-simplex"
        });
        g.setDefaultEdgeLabel(() => ({}));

        // Add nodes
        elements.filter(e => !e.data.source).forEach(n => {
          g.setNode(n.data.id, {
            width: n.data.width || 100,
            height: 34,
            label: n.data.label || ""
          });
          if (n.data.parent) {
            g.setParent(n.data.id, n.data.parent);
          }
        });

        // Add edges
        elements.filter(e => e.data.source).forEach(e => {
          g.setEdge(e.data.source, e.data.target);
        });

        dagre.layout(g);

        const positions = {};
        g.nodes().forEach(id => {
          const node = g.node(id);
          if (node) {
            positions[id] = { x: node.x, y: node.y };
          }
        });

        self.postMessage({ type: "layout-result", positions });
      } else {
        self.postMessage({ type: "layout-error", error: "dagre not available in worker" });
      }
    } catch (err) {
      self.postMessage({ type: "layout-error", error: err.message });
    }
  }
};
