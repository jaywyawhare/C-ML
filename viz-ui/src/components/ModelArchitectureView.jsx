import React, { useEffect, useMemo, useRef } from 'react';
import cytoscape from 'cytoscape';
import elk from 'cytoscape-elk';

elk(cytoscape);

function parseModelSummary(summary) {
  if (!summary || summary === 'N/A') return [];

  // Try to parse as JSON first (new C-ML architecture format)
  try {
    const parsed = typeof summary === 'string' ? JSON.parse(summary) : summary;
    if (parsed && parsed.layers && Array.isArray(parsed.layers)) {
      return parsed.layers.map(layer => ({
        name: layer.type || layer.name || 'Unknown',
        params: {
          in_features: layer.in_features,
          out_features: layer.out_features,
          in_channels: layer.in_channels,
          out_channels: layer.out_channels,
          kernel_size: layer.kernel_size,
          stride: layer.stride,
          padding: layer.padding,
          has_bias: layer.has_bias,
          num_params: layer.num_params,
        },
      }));
    }
  } catch (e) {
    // Not JSON, continue with string parsing
  }

  // Parse strings like "Sequential(Linear(2->4), ReLU, Linear(4->4), Tanh, Linear(4->1), Sigmoid)"
  const layers = [];
  let current = typeof summary === 'string' ? summary.trim() : String(summary);

  // Remove "Sequential(" wrapper if present
  if (current.startsWith('Sequential(')) {
    current = current.slice(11, -1); // Remove "Sequential(" and last ")"
  }

  // Split by comma, but be careful with nested parentheses
  let depth = 0;
  let start = 0;

  for (let i = 0; i < current.length; i++) {
    const char = current[i];
    if (char === '(') depth++;
    else if (char === ')') depth--;
    else if (char === ',' && depth === 0) {
      const layerStr = current.slice(start, i).trim();
      if (layerStr) {
        layers.push(parseLayer(layerStr));
      }
      start = i + 1;
    }
  }

  // Add last layer
  const lastLayer = current.slice(start).trim();
  if (lastLayer) {
    layers.push(parseLayer(lastLayer));
  }

  return layers;
}

function parseLayer(layerStr) {
  layerStr = layerStr.trim();

  // Extract layer name (before first parenthesis)
  const parenIdx = layerStr.indexOf('(');
  const name = parenIdx > 0 ? layerStr.slice(0, parenIdx).trim() : layerStr;

  // Extract parameters if present
  let params = {};
  if (parenIdx > 0) {
    const paramsStr = layerStr.slice(parenIdx + 1, -1);

    // Check for input->output pattern (e.g., "2->4")
    const arrowMatch = paramsStr.match(/(\d+)\s*->\s*(\d+)/);
    if (arrowMatch) {
      params.inputSize = parseInt(arrowMatch[1]);
      params.outputSize = parseInt(arrowMatch[2]);
    } else {
      // Try to extract any numbers
      const numbers = paramsStr.match(/\d+/g);
      if (numbers) {
        if (numbers.length >= 2) {
          params.inputSize = parseInt(numbers[0]);
          params.outputSize = parseInt(numbers[1]);
        } else if (numbers.length === 1) {
          params.size = parseInt(numbers[0]);
        }
      }
    }
  }

  return { name, params };
}

function getLayerColor(layerName) {
  const name = layerName.toLowerCase();
  // Using consistent, vibrant colors matching Ops Topology
  if (name.includes('relu')) {
    return '#ec4899'; // Pink for ReLU
  } else if (name.includes('linear')) {
    return '#3b82f6'; // Blue for Linear
  } else if (name.includes('conv')) {
    return '#8b5cf6'; // Purple for Conv
  } else if (name.includes('pool') || name.includes('maxpool') || name.includes('avgpool') || name.includes('lrn')) {
    return '#14b8a6'; // Teal for Pooling
  } else if (name.includes('concat')) {
    return '#f59e0b'; // Orange for Concat
  } else if (name.includes('tanh')) {
    return '#a855f7'; // Purple for Tanh
  } else if (name.includes('sigmoid')) {
    return '#f472b6'; // Pink for Sigmoid
  } else if (name.includes('softmax')) {
    return '#eab308'; // Yellow for Softmax
  } else if (name.includes('batchnorm') || name.includes('layernorm')) {
    return '#06b6d4'; // Cyan for Normalization
  } else if (name.includes('dropout')) {
    return '#f43f5e'; // Rose for Dropout
  }
  return '#64748b'; // Default gray
}

function toCytoscapeElements(layers) {
  if (!layers || layers.length === 0) return [];

  const elements = [];

  // Input node
  elements.push({
    data: {
      id: 'input',
      label: 'Input',
      color: '#10b981',
      type: 'io'
    }
  });

  // Layer nodes
  layers.forEach((layer, idx) => {
    const color = getLayerColor(layer.name);
    let label = layer.name;
    let details = '';

    // Handle different layer types with appropriate details
    if (layer.params.in_features && layer.params.out_features) {
      details = `${layer.params.in_features} → ${layer.params.out_features}`;
    } else if (layer.params.in_channels && layer.params.out_channels) {
      details = `${layer.params.in_channels}ch → ${layer.params.out_channels}ch`;
      if (layer.params.kernel_size) {
        details += ` (k${layer.params.kernel_size})`;
      }
    } else if (layer.params.inputSize && layer.params.outputSize) {
      details = `${layer.params.inputSize} → ${layer.params.outputSize}`;
    } else if (layer.params.size) {
      details = `size: ${layer.params.size}`;
    } else if (layer.params.kernel_size) {
      details = `kernel: ${layer.params.kernel_size}`;
    }

    // Add parameter count if available
    if (layer.params.num_params && layer.params.num_params > 0) {
      const paramStr = layer.params.num_params >= 1000
        ? `${(layer.params.num_params / 1000).toFixed(1)}K`
        : String(layer.params.num_params);
      details = details ? `${details} (${paramStr} params)` : `${paramStr} params`;
    }

    // Combine label and details
    const fullLabel = details ? `${label}\n${details}` : label;

    elements.push({
      data: {
        id: `layer-${idx}`,
        label: fullLabel,
        color: color,
        type: 'layer'
      }
    });

    // Edge from previous node
    const sourceId = idx === 0 ? 'input' : `layer-${idx - 1}`;
    elements.push({
      data: {
        id: `${sourceId}->layer-${idx}`,
        source: sourceId,
        target: `layer-${idx}`
      }
    });
  });

  // Output node
  elements.push({
    data: {
      id: 'output',
      label: 'Output',
      color: '#10b981',
      type: 'io'
    }
  });

  // Edge from last layer to output
  elements.push({
    data: {
      id: `layer-${layers.length - 1}->output`,
      source: `layer-${layers.length - 1}`,
      target: 'output'
    }
  });

  return elements;
}

export default function ModelArchitectureView({ modelSummary }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const layers = useMemo(() => parseModelSummary(modelSummary), [modelSummary]);
  const elements = useMemo(() => toCytoscapeElements(layers), [layers]);

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
          selector: 'node[type="layer"]',
          style: {
            'shape': 'round-rectangle',
            'background-color': 'transparent',
            'background-opacity': 0,
            'border-color': 'data(color)',
            'border-width': 3,
            'label': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': '280px',
            'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
            'font-size': 26,
            'color': '#fff',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 320,
            'height': 90,
            'padding': '20px 28px',
            'min-width': 280,
            'min-height': 80,
            'font-weight': 'normal'
          }
        },
        {
          selector: 'node[type="io"]',
          style: {
            'shape': 'round-rectangle',
            'background-color': 'transparent',
            'background-opacity': 0,
            'border-color': 'data(color)',
            'border-width': 3,
            'label': 'data(label)',
            'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
            'font-size': 26,
            'color': '#f9fafb',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 320,
            'height': 90,
            'padding': '20px 28px',
            'min-width': 280,
            'min-height': 80,
            'font-weight': 'normal'
          }
        },
        {
          selector: 'edge',
          style: {
            'curve-style': 'bezier',
            'line-color': '#94a3b8',
            'target-arrow-color': '#94a3b8',
            'target-arrow-shape': 'triangle',
            'width': 2.5,
            'opacity': 0.7,
            'line-style': 'solid'
          }
        }
      ],
      layout: {
        name: 'elk',
        elk: {
          algorithm: 'layered',
          'elk.direction': 'RIGHT',
          'elk.edgeRouting': 'ORTHOGONAL',
          'spacing.nodeNodeBetweenLayers': '50',
          'spacing.edgeEdgeBetweenLayers': '20',
          'spacing.nodeNode': '30',
          'layered.spacing.nodeNodeBetweenLayers': '50',
          'layered.spacing.edgeNodeBetweenLayers': '20',
          'layered.spacing.edgeEdgeBetweenLayers': '15',
          'elk.layered.nodePlacement.strategy': 'SIMPLE',
          'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
          'elk.layered.cycleBreaking.strategy': 'GREEDY',
          'elk.layered.compaction.postCompaction': 'true',
          'elk.layered.nodePlacement.bk.fixedAlignment': 'LEFTUP',
          'elk.padding': '[top=40,left=40,bottom=40,right=40]'
        },
        nodeDimensionsIncludeLabels: true,
        fit: true,
        padding: 40
      },
      minZoom: 0.1,
      maxZoom: 3
    });

    const cy = cyRef.current;

    cy.once('layoutstop', () => {
      // Fit the graph to the container with some padding
      cy.fit(undefined, 16);
    });

    cy.on('tap', 'node', (evt) => {
      const n = evt.target;
      cy.animate({ center: { eles: n }, zoom: Math.min(cy.zoom() * 1.1, 2.5) }, { duration: 180 });
    });

    return () => {
      cy.destroy();
    };
  }, [elements]);

  if (layers.length === 0) {
    return (
      <div style={{ padding: 24, textAlign: 'center', color: 'var(--muted)' }}>
        No model architecture available
      </div>
    );
  }

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: 'transparent', overflow: 'auto' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
