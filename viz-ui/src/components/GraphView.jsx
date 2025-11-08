import React, { useEffect, useMemo, useRef } from 'react';
import cytoscape from 'cytoscape';
import elk from 'cytoscape-elk';
import ModelArchitectureView from './ModelArchitectureView.jsx';

elk(cytoscape);

function toCytoscapeElements(graph) {
  if (!graph) return [];
  const elements = [];
  for (const [id, node] of Object.entries(graph)) {
    elements.push({ data: { id: String(id), label: node.label || String(id), color: node.color || '#e0e0e0' } });
  }
  for (const [toId, node] of Object.entries(graph)) {
    const src = node.src || [];
    for (const [, fromId] of src) {
      if (graph[fromId] !== undefined) {
        elements.push({ data: { id: `${fromId}->${toId}`, source: String(fromId), target: String(toId) } });
      }
    }
  }
  return elements;
}

export default function GraphView({ graph, training }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const elements = useMemo(() => toCytoscapeElements(graph), [graph]);

  useEffect(() => {
    if (!containerRef.current) return;
    if (cyRef.current) {
      cyRef.current.destroy();
      cyRef.current = null;
    }

    cyRef.current = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'shape': 'round-rectangle',
            'background-color': 'data(color)',
            'border-color': '#1f2937',
            'border-width': 1,
            'label': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': '280px',
            'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
            'font-size': 14,
            'color': '#111827',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 280,
            'height': 72,
            'padding': '20px 30px',
            'min-width': 220,
            'min-height': 72
          }
        },
        {
          selector: 'edge',
          style: {
            'curve-style': 'bezier',
            'line-color': '#94a3b8',
            'target-arrow-color': '#94a3b8',
            'target-arrow-shape': 'triangle',
            'width': 1.5,
            'line-style': 'solid'
          }
        }
      ],
      layout: {
        name: 'elk',
        elk: {
          algorithm: 'layered',
          'elk.direction': 'DOWN',
          'elk.edgeRouting': 'ORTHOGONAL',
          'spacing.nodeNodeBetweenLayers': '28',
          'spacing.edgeEdgeBetweenLayers': '12',
          'spacing.nodeNode': '16',
          'layered.spacing.nodeNodeBetweenLayers': '28',
          'layered.spacing.edgeNodeBetweenLayers': '12',
          'layered.spacing.edgeEdgeBetweenLayers': '8',
          'elk.layered.nodePlacement.strategy': 'SIMPLE',
          'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
          'elk.layered.cycleBreaking.strategy': 'GREEDY',
          'elk.layered.compaction.postCompaction': 'true',
          'elk.layered.nodePlacement.bk.fixedAlignment': 'LEFTUP',
          'elk.padding': '[top=16,left=16,bottom=16,right=16]'
        },
        nodeDimensionsIncludeLabels: true,
        fit: false,
        padding: 24
      },
      minZoom: 0.1,
      maxZoom: 3
    });

    const cy = cyRef.current;

    cy.once('layoutstop', () => {
      const container = containerRef.current;
      if (!container) return;
      const zoom = cy.zoom();
      const extent = cy.extent();
      const graphWidth = extent.x2 - extent.x1;
      const graphHeight = extent.y2 - extent.y1;
      const panX = 40 - extent.x1 * zoom;
      const panY = (container.clientHeight - graphHeight * zoom) / 2 - extent.y1 * zoom;
      cy.pan({
        x: panX,
        y: panY,
      });
    });

    cy.on('tap', 'node', (evt) => {
      const n = evt.target;
      cy.animate({ center: { eles: n }, zoom: Math.min(cy.zoom() * 1.1, 2.5) }, { duration: 180 });
    });

    return () => {
      cy.destroy();
    };
  }, [elements]);

  const modelSummary = training?.model_summary || null;

  return (
    <div
      style={{
        padding: 16,
        height: '100%',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        background:
          'radial-gradient(circle at 15% 20%, rgba(99,102,241,0.08), transparent 55%), radial-gradient(circle at 85% 25%, rgba(16,185,129,0.08), transparent 50%)',
      }}
    >
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: modelSummary ? 'minmax(0, 1fr) minmax(0, 0.9fr)' : 'minmax(0, 1fr)',
          gap: 16,
          flex: 1,
          minHeight: 0,
        }}
      >
        <div
          style={{
            position: 'relative',
            padding: 18,
            borderRadius: 18,
            border: '1px solid rgba(148,163,184,0.18)',
            background: 'linear-gradient(135deg, rgba(15,23,42,0.75), rgba(30,41,59,0.6))',
            boxShadow: '0 18px 45px rgba(15,23,42,0.35)',
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
            minHeight: 0,
            overflow: 'auto'
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ margin: 0, color: 'var(--text)', fontSize: 15, fontWeight: 600 }}>Ops Topology</h4>
          </div>
          <div
            style={{
              position: 'relative',
              flex: 1,
              minHeight: 0,
              borderRadius: 14,
              overflow: 'auto',
              border: '1px solid rgba(148,163,184,0.15)',
              background: 'radial-gradient(circle at 20% 20%, rgba(148,163,184,0.08), transparent 70%)',
            }}
          >
            <div ref={containerRef} style={{ position: 'relative', minWidth: '100%', minHeight: '100%' }} />
          </div>
        </div>

        {modelSummary ? (
          <div
            style={{
              position: 'relative',
              padding: 18,
              borderRadius: 18,
              border: '1px solid rgba(148,163,184,0.18)',
              background: 'linear-gradient(135deg, rgba(30,41,59,0.75), rgba(15,23,42,0.65))',
              boxShadow: '0 18px 45px rgba(15,23,42,0.32)',
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
              minHeight: 0,
              overflow: 'auto'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h4 style={{ margin: 0, color: 'var(--text)', fontSize: 15, fontWeight: 600 }}>Model Architecture</h4>
            </div>
            <div
              style={{
                flex: 1,
                minHeight: 0,
                borderRadius: 14,
                overflow: 'auto',
                border: '1px solid rgba(148,163,184,0.15)',
                background: 'radial-gradient(circle at 80% 20%, rgba(99,102,241,0.1), transparent 70%)',
              }}
            >
              <ModelArchitectureView modelSummary={modelSummary} />
            </div>
          </div>
        ) : (
          <div
            style={{
              position: 'relative',
              padding: 18,
              borderRadius: 18,
              border: '1px dashed rgba(148,163,184,0.25)',
              background: 'linear-gradient(135deg, rgba(15,23,42,0.6), rgba(30,41,59,0.55))',
              boxShadow: '0 12px 35px rgba(15,23,42,0.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'rgba(226,232,240,0.65)',
              fontSize: 13,
              letterSpacing: 0.3,
            }}
          >
            Awaiting model summary export…
          </div>
        )}
      </div>
    </div>
  );
}
