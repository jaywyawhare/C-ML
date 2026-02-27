import React, { memo, useEffect, useMemo, useRef, useState, useCallback } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import cytoscapeElk from 'cytoscape-elk';
import { Box, Layers, Cpu, Activity, Zap, Maximize2, Minimize2, X, GitBranch } from 'lucide-react';
import ModelArchitectureView from './ModelArchitectureView.jsx';

cytoscape.use(cytoscapeElk);
cytoscape.use(dagre);

// Modern color palette
const getNodeColor = (label, isDead, isFused) => {
  if (isDead) return '#ef4444'; // Red for Dead Code
  if (isFused) return '#10b981'; // Emerald for Fused Kernel

  if (!label) return '#71717a';
  const name = label.toLowerCase();

  if (name.includes('relu') || name.includes('sigmoid') || name.includes('tanh')) {
    return '#ec4899'; // Pink
  } else if (name.includes('conv')) {
    return '#8b5cf6'; // Violet
  } else if (name.includes('pool')) {
    return '#06b6d4'; // Cyan
  } else if (name.includes('loss')) {
    return '#f43f5e'; // Rose
  }
  return '#64748b'; // Slate (default for all other ops including MATMUL)
};

function toCytoscapeElements(graph, kernels) {
  if (!graph) return [];
  const elements = [];
  const visibleNodes = new Set();
  const fusedGroups = {};
  const usedAsSource = new Set();

  // First pass: Identify fused groups, visible nodes, and nodes used as sources
  for (const [id, node] of Object.entries(graph)) {
    visibleNodes.add(String(id));
    // Filter out NULL pointers (exported as "(nil)" or "0x0")
    const hasValidKernelId = node.fusedKernelId &&
      node.fusedKernelId !== '(nil)' &&
      node.fusedKernelId !== '0x0';
    if (node.is_fused && hasValidKernelId) {
      if (!fusedGroups[node.fusedKernelId]) {
        fusedGroups[node.fusedKernelId] = [];
      }
      fusedGroups[node.fusedKernelId].push(id);
    }

    // Track usage as source to identify output nodes
    const src = node.src || [];
    for (const edge of src) {
      const fromId = Array.isArray(edge) ? edge[1] : edge;
      usedAsSource.add(String(fromId));
    }
  }

  // Create parent nodes for fused kernels
  for (const [kernelId, nodeIds] of Object.entries(fusedGroups)) {
    // Check if ALL nodes in the kernel are dead to mark the kernel as dead
    const allDead = nodeIds.every(id => graph[id].is_dead);

    elements.push({
      data: {
        id: kernelId,
        label: '', // No label for the box itself
        isFusedGroup: true,
        isDead: allDead
      },
      classes: 'fused-cluster' + (allDead ? ' dead' : '')
    });
  }

  // Create nodes
  for (const [id, node] of Object.entries(graph)) {
    const isDead = node.is_dead;
    const isFused = node.is_fused;
    const fusedKernelId = node.fusedKernelId;
    const label = node.label || String(id);
    const color = node.color || getNodeColor(label, isDead, isFused);

    const width = Math.max(80, label.length * 9 + 40);

    elements.push({
      data: {
        id: String(id),
        label: label,
        color: color,
        isDead: isDead,
        isFused: isFused,
        width: width,
        parent: (isFused && fusedKernelId) ? fusedKernelId : undefined
      },
      classes: (isDead ? 'dead ' : '') + (isFused ? 'fused' : '')
    });
  }

  // Create edges
  for (const [toId, node] of Object.entries(graph)) {
    if (!visibleNodes.has(String(toId))) continue;
    const src = node.src || [];
    for (const edge of src) {
      // Handle both [index, id] format and simple id format
      let fromId, inputIndex;
      if (Array.isArray(edge)) {
        inputIndex = edge[0];
        fromId = edge[1];
      } else {
        fromId = edge;
      }

      if (!visibleNodes.has(String(fromId))) continue;

      // Determine if this is an internal edge within a fused kernel
      const fromNode = graph[fromId];
      const toNode = graph[toId];

      // Strict check: Both must be fused AND share the exact same kernel ID
      const isInternal = fromNode.is_fused && toNode.is_fused &&
        fromNode.fusedKernelId && toNode.fusedKernelId &&
        fromNode.fusedKernelId === toNode.fusedKernelId;

      elements.push({
        data: {
          id: `${fromId}->${toId}`,
          source: String(fromId),
          target: String(toId),
          label: inputIndex !== undefined ? String(inputIndex) : '', // Show input index
          isInternal: isInternal
        },
        classes: isInternal ? 'internal-edge' : ''
      });
    }
  }

  return elements;
}


const GraphView = memo(function GraphView({ graph, training, modelArch, kernels }) {
  const [activeView, setActiveView] = useState('ops');
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  const elements = useMemo(() => toCytoscapeElements(graph, kernels), [graph, kernels]);

  // Calculate dynamic spacing based on node count
  const nodeCount = graph ? Object.keys(graph).length : 0;
  const dynamicSpacing = useMemo(() => {
    // Scale spacing based on number of nodes - more compact for larger graphs
    const baseNodeSep = nodeCount < 10 ? 70 : nodeCount < 20 ? 50 : 35;
    const baseRankSep = nodeCount < 10 ? 90 : nodeCount < 20 ? 65 : 50;

    return {
      nodeSep: baseNodeSep,
      rankSep: baseRankSep,
      edgeSep: Math.max(25, 50 - nodeCount),
      padding: 35
    };
  }, [nodeCount]);

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;

    // If instance exists, just update elements and layout
    if (cyRef.current) {
      cyRef.current.json({ elements });
      cyRef.current.layout({
        name: 'dagre',
        rankDir: 'LR',
        ranker: 'network-simplex',
        ...dynamicSpacing,
        fit: true,
        animate: true,
        animationDuration: 300
      }).run();
      return;
    }

    // Create new instance - use fit:false initially, then fit after layout
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'shape': 'round-rectangle',
            'background-color': '#18181b',
            'border-color': 'data(color)',
            'border-width': 1,
            'label': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': '120px',
            'font-family': 'Inter, system-ui, sans-serif',
            'font-size': 10,
            'font-weight': 500,
            'color': '#e4e4e7',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 'data(width)',
            'height': 34,
            'padding': '5px',
            'text-margin-y': 0,
            'transition-property': 'background-color, border-width, border-color, width, height',
            'transition-duration': '0.2s',
            'ghost': 'yes',
            'ghost-offset-x': 0,
            'ghost-offset-y': 2,
            'ghost-opacity': 0.1
          }
        },
        {
          selector: 'node.fused-cluster',
          style: {
            'background-color': 'transparent',
            'background-opacity': 0,
            'border-color': '#10b981',
            'border-width': 2,
            'border-style': 'dashed',
            'label': '',
            'shape': 'round-rectangle',
            'padding': 12
          }
        },
        {
          selector: 'node.dead',
          style: {
            'border-color': '#ef4444',
            'border-style': 'dashed',
            'border-width': 2,
            'background-color': '#18181b',
            'color': '#a1a1aa'
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#fff',
            'border-width': 2,
            'background-color': '#27272a',
            'z-index': 999
          }
        },
        {
          selector: 'edge',
          style: {
            'curve-style': 'taxi',
            'taxi-direction': 'right',
            'taxi-turn': 15,
            'taxi-turn-min-distance': 8,
            'line-color': '#52525b',
            'line-style': 'solid',
            'width': 1.5,
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#52525b',
            'arrow-scale': 0.9,
            'font-size': 9,
            'color': '#a1a1aa',
            'text-background-color': '#18181b',
            'text-background-opacity': 1,
            'text-background-padding': 2,
            'text-rotation': 'autorotate',
            'transition-property': 'line-color, width, line-style',
            'transition-duration': '0.1s'
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#fafafa',
            'target-arrow-color': '#fafafa',
            'width': 2,
            'z-index': 999
          }
        },
        {
          selector: 'node.dimmed',
          style: {
            'opacity': 0.75,
            'z-index': 1,
            'transition-duration': '0.1s'
          }
        },
        {
          selector: 'edge.dimmed',
          style: {
            'opacity': 0.75,
            'z-index': 1,
            'transition-duration': '0.1s'
          }
        },
        {
          selector: '.highlighted',
          style: {
            'background-color': '#3b82f6',
            'line-color': '#3b82f6',
            'target-arrow-color': '#3b82f6',
            'border-color': '#60a5fa',
            'border-width': 2,
            'z-index': 9999,
            'transition-duration': '0.1s'
          }
        },
        {
          selector: 'edge.highlighted',
          style: {
            'width': 3,
            'line-style': 'solid'
          }
        },
        {
          selector: 'edge.incoming-highlighted',
          style: {
            'line-color': '#f59e0b',
            'target-arrow-color': '#f59e0b',
            'width': 3,
            'line-style': 'dashed',
            'z-index': 9999
          }
        },
        {
          selector: 'edge.outgoing-highlighted',
          style: {
            'line-color': '#a855f7',
            'target-arrow-color': '#a855f7',
            'width': 3,
            'line-style': 'solid',
            'z-index': 9999
          }
        },
        {
          selector: 'node.incoming-highlighted',
          style: {
            'border-color': '#f59e0b',
            'border-width': 3
          }
        },
        {
          selector: 'node.outgoing-highlighted',
          style: {
            'border-color': '#a855f7',
            'border-width': 3
          }
        },
        {
          selector: '.internal-edge',
          style: {
            'line-color': '#71717a',
            'target-arrow-color': '#71717a',
            'width': 1.5,
            'line-style': 'solid',
            'z-index': 10
          }
        }
      ],
      layout: {
        name: 'dagre',
        rankDir: 'LR',
        ranker: 'network-simplex',
        ...dynamicSpacing,
        fit: true,
        animate: false  // No animation on initial render
      },
      minZoom: 0.2,
      maxZoom: 3,
      autoungrabify: true,
      boxSelectionEnabled: false
    });

    const cy = cyRef.current;

    const resetHighlights = () => {
      cy.elements().removeClass('highlighted incoming-highlighted outgoing-highlighted dimmed');
    };

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        resetHighlights();
        return;
      }

      const target = evt.target;
      resetHighlights();

      // Dim everything else by 0.2 (opacity 0.8)
      cy.elements().addClass('dimmed');

      // Helper to highlight a node and ensure its parent is not dimmed
      const highlightNode = (node) => {
        node.addClass('highlighted').removeClass('dimmed');
        if (node.isChild()) {
          // Do not highlight the parent, just ensure it's not dimmed
          node.parent().removeClass('dimmed');
        }
      };

      if (target.isEdge()) {
        // Highlight edge and connected nodes
        target.addClass('highlighted').removeClass('dimmed');
        highlightNode(target.source());
        highlightNode(target.target());
      } else if (target.isNode()) {
        // Highlight node
        highlightNode(target);

        // Incoming Edges (Dashed Orange)
        const incomers = target.incomers('edge');
        incomers.addClass('incoming-highlighted').removeClass('dimmed');
        incomers.sources().forEach(source => {
          source.addClass('incoming-highlighted').removeClass('dimmed');
          if (source.isChild()) {
            // Do not highlight the parent, just ensure it's not dimmed
            source.parent().removeClass('dimmed');
          }
        });

        // Outgoing Edges (Solid Purple)
        const outgoers = target.outgoers('edge');
        outgoers.addClass('outgoing-highlighted').removeClass('dimmed');
        outgoers.targets().forEach(target => {
          target.addClass('outgoing-highlighted').removeClass('dimmed');
          if (target.isChild()) {
            // Do not highlight the parent, just ensure it's not dimmed
            target.parent().removeClass('dimmed');
          }
        });
      }
    });

    cy.on('tap', 'node', (evt) => {
      // Keep the zoom animation but let the highlight logic above handle styles
      cy.animate({
        center: { eles: evt.target },
        zoom: 1.2
      }, { duration: 300 });
    });

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [elements, dynamicSpacing]); // Re-run when elements or spacing changes

  // Debounced layout update to prevent excessive re-renders
  const layoutTimeoutRef = useRef(null);
  const lastElementsRef = useRef(null);

  // Handle updates when elements change (debounced)
  useEffect(() => {
    // Skip if elements haven't actually changed
    const elementsStr = JSON.stringify(elements);
    if (elementsStr === lastElementsRef.current) return;
    lastElementsRef.current = elementsStr;

    if (layoutTimeoutRef.current) {
      clearTimeout(layoutTimeoutRef.current);
    }

    layoutTimeoutRef.current = setTimeout(() => {
      if (cyRef.current) {
        cyRef.current.json({ elements });
        // Use faster dagre layout for updates, elk only for initial
        cyRef.current.layout({
          name: 'dagre',
          rankDir: 'LR',
          ranker: 'network-simplex',
          ...dynamicSpacing,
          fit: true,
          animate: true,
          animationDuration: 200
        }).run();
      }
    }, 150); // Debounce layout updates

    return () => {
      if (layoutTimeoutRef.current) {
        clearTimeout(layoutTimeoutRef.current);
      }
    };
  }, [elements, dynamicSpacing]);

  // Handle visibility changes (resize)
  useEffect(() => {
    if (activeView === 'ops' && cyRef.current) {
      // Small delay to allow container to become visible
      requestAnimationFrame(() => {
        cyRef.current.resize();
        cyRef.current.fit(undefined, 40);
      });
    }
  }, [activeView]);

  const modelSummary = modelArch || training?.model_summary || null;

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg-dark)',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        height: 60,
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid var(--border-color)',
        background: 'rgba(24, 24, 27, 0.5)',
        backdropFilter: 'blur(12px)',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', gap: 4, background: 'rgba(255, 255, 255, 0.03)', padding: 4, borderRadius: 8, border: '1px solid var(--border-color)' }}>
          <ViewTab
            active={activeView === 'ops'}
            onClick={() => setActiveView('ops')}
            label="Ops Topology"
            icon={<GitBranch size={14} />}
            count={graph ? Object.keys(graph).length : 0}
          />
          {modelSummary && (
            <ViewTab
              active={activeView === 'model'}
              onClick={() => setActiveView('model')}
              label="Architecture"
              icon={<Cpu size={14} />}
            />
          )}
        </div>

        {/* Legend - Only visible in Ops Topology */}
        {activeView === 'ops' && (
          <div style={{ display: 'flex', gap: 16 }}>
            <LegendItem color="#ef4444" label="Dead Code" dashed />
            <LegendItem color="#10b981" label="Fused Kernel" dashed />
            <LegendItem color="#f59e0b" label="Input Flow" />
            <LegendItem color="#a855f7" label="Output Flow" />
            <LegendItem color="#3b82f6" label="Selected" />
          </div>
        )}
      </div>

      {/* Content */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>

        {/* Graph Container - Always rendered, hidden when inactive */}
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          visibility: activeView === 'ops' ? 'visible' : 'hidden',
          zIndex: 1
        }}>
          {!graph || Object.keys(graph).length === 0 ? (
            <div style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--text-tertiary)',
              gap: 16
            }}>
              <Box size={48} strokeWidth={1} />
              <div style={{ fontSize: 16 }}>Waiting for graph data...</div>
            </div>
          ) : (
            <>
              <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
            </>
          )}
        </div>

        {/* Model Architecture View */}
        {activeView === 'model' && (
          <div style={{
            position: 'absolute',
            top: 0, left: 0, right: 0, bottom: 0,
            zIndex: 2,
            background: 'var(--bg-dark)',
            padding: 24
          }}>
            <div style={{ width: '100%', height: '100%', overflow: 'hidden' }}>
              <ModelArchitectureView modelSummary={modelSummary} />
            </div>
          </div>
        )}

      </div>
    </div>
  );
});

export default GraphView;

const ViewTab = memo(function ViewTab({ active, onClick, label, icon, count }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '6px 12px',
        borderRadius: 6,
        border: 'none',
        background: active ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
        color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
        fontSize: 13,
        fontWeight: 500,
        cursor: 'pointer',
        transition: 'all 0.2s'
      }}
    >
      {icon}
      {label}
      {count !== undefined && count > 0 && (
        <span style={{
          background: 'rgba(255, 255, 255, 0.1)',
          padding: '1px 6px',
          borderRadius: 4,
          fontSize: 11
        }}>
          {count}
        </span>
      )}
    </button>
  );
});

const LegendItem = memo(function LegendItem({ color, label, dashed }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        width: 10,
        height: 10,
        borderRadius: '50%',
        background: dashed ? 'transparent' : color,
        border: dashed ? `2px dashed ${color}` : 'none'
      }} />
      <span style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 500 }}>{label}</span>
    </div>
  );
});
