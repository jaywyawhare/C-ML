import React, { useState, useMemo, memo } from 'react';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import c from 'react-syntax-highlighter/dist/esm/languages/prism/c';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  Cpu,
  Zap,
  Trash2,
  Layers,
  Code2,
  Activity,
  Box,
  Terminal,
  FileCode,
  Server,
  Monitor,
  Smartphone
} from 'lucide-react';

SyntaxHighlighter.registerLanguage('c', c);

// Helper: Transpile a single operation to target backend
function transpileOp(op, backend, index) {
  const { type, inputs, output } = op;
  const i = index !== undefined ? index : '';

  // Helper to get input variable names
  const in0 = inputs[0] || 'in0';
  const in1 = inputs[1] || 'in1';
  const in2 = inputs[2] || 'in2';
  const out = output || `t${i}`;

  switch (backend) {
    case 'cuda':
    case 'opencl':
    case 'metal':
    case 'wgsl':
      // GPU-like syntax
      switch (type) {
        case 'ADD': return `${out} = ${in0} + ${in1};`;
        case 'SUB': return `${out} = ${in0} - ${in1};`;
        case 'MUL': return `${out} = ${in0} * ${in1};`;
        case 'DIV': return `${out} = ${in0} / ${in1};`;
        case 'NEG': return `${out} = -${in0};`;
        case 'RECIP': return `${out} = 1.0f / ${in0};`;
        case 'EXP': return `${out} = exp(${in0});`;
        case 'LOG': return `${out} = log(${in0});`;
        case 'SQRT': return `${out} = sqrt(${in0});`;
        case 'ABS': return `${out} = abs(${in0});`;
        case 'SIN': return `${out} = sin(${in0});`;
        case 'COS': return `${out} = cos(${in0});`;
        case 'MAX': return `${out} = max(${in0}, ${in1});`;
        case 'MIN': return `${out} = min(${in0}, ${in1});`;
        case 'RELU': return `${out} = max(${in0}, 0.0f);`;
        case 'SIGMOID':
        case 'UNKNOWN': // UNKNOWN is often sigmoid
          return `${out} = 1.0f / (1.0f + exp(-${in0}));`;
        case 'TANH': return `${out} = tanh(${in0});`;
        case 'MATMUL': return `// Matrix Multiplication\n    ${out} = 0.0f;\n    for (int k = 0; k < K; ++k) {\n        ${out} += ${in0}[row * K + k] * ${in1}[k * N + col];\n    }`;
        case 'MEAN': return `// Reduction: Mean\n    float sum = 0.0f;\n    for (int j = 0; j < n; j++) sum += ${in0}[j];\n    ${out} = sum / (float)n;`;
        case 'SUM': return `// Reduction: Sum\n    float sum = 0.0f;\n    for (int j = 0; j < n; j++) sum += ${in0}[j];\n    ${out} = sum;`;
        case 'FILL': return `${out} = 0.0f; // Constant fill`;
        case 'PERMUTE':
        case 'RESHAPE':
        case 'EXPAND':
        case 'SLICE':
        case 'STRIDE':
          return `// View operation: ${type} (no computation, memory layout change)`;
        default: return `// ${type}: Custom operation`;
      }
    case 'c_simd':
      // AVX2 intrinsics (simplified)
      switch (type) {
        case 'ADD': return `_mm256_store_ps(${out}, _mm256_add_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`;
        case 'SUB': return `_mm256_store_ps(${out}, _mm256_sub_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`;
        case 'MUL': return `_mm256_store_ps(${out}, _mm256_mul_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`;
        case 'DIV': return `_mm256_store_ps(${out}, _mm256_div_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`;
        case 'MAX': return `_mm256_store_ps(${out}, _mm256_max_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`;
        case 'MIN': return `_mm256_store_ps(${out}, _mm256_min_ps(_mm256_load_ps(${in0}), _mm256_load_ps(${in1})));`;
        case 'SQRT': return `_mm256_store_ps(${out}, _mm256_sqrt_ps(_mm256_load_ps(${in0})));`;
        default: return `// ${type}: Scalar fallback needed for SIMD`;
      }
    case 'c':
    case 'cpu':
    default:
      // Standard C
      switch (type) {
        case 'ADD': return `${out}[i] = ${in0}[i] + ${in1}[i];`;
        case 'SUB': return `${out}[i] = ${in0}[i] - ${in1}[i];`;
        case 'MUL': return `${out}[i] = ${in0}[i] * ${in1}[i];`;
        case 'DIV': return `${out}[i] = ${in0}[i] / ${in1}[i];`;
        case 'NEG': return `${out}[i] = -${in0}[i];`;
        case 'RECIP': return `${out}[i] = 1.0f / ${in0}[i];`;
        case 'EXP': return `${out}[i] = expf(${in0}[i]);`;
        case 'LOG': return `${out}[i] = logf(${in0}[i]);`;
        case 'SQRT': return `${out}[i] = sqrtf(${in0}[i]);`;
        case 'ABS': return `${out}[i] = fabsf(${in0}[i]);`;
        case 'SIN': return `${out}[i] = sinf(${in0}[i]);`;
        case 'COS': return `${out}[i] = cosf(${in0}[i]);`;
        case 'MAX': return `${out}[i] = fmaxf(${in0}[i], ${in1}[i]);`;
        case 'MIN': return `${out}[i] = fminf(${in0}[i], ${in1}[i]);`;
        case 'RELU': return `${out}[i] = fmaxf(${in0}[i], 0.0f);`;
        case 'SIGMOID':
        case 'UNKNOWN': // UNKNOWN is often sigmoid
          return `${out}[i] = 1.0f / (1.0f + expf(-${in0}[i]));`;
        case 'TANH': return `${out}[i] = tanhf(${in0}[i]);`;
        case 'MATMUL': return `for (int m = 0; m < M; m++) {\n        for (int n = 0; n < N; n++) {\n            float sum = 0.0f;\n            for (int k = 0; k < K; k++) sum += ${in0}[m*K+k] * ${in1}[k*N+n];\n            ${out}[m*N+n] = sum;\n        }\n    }`;
        case 'MEAN': return `float sum = 0.0f;\n        for (int j = 0; j < n; j++) sum += ${in0}[j];\n        ${out}[0] = sum / (float)n;`;
        case 'SUM': return `float sum = 0.0f;\n        for (int j = 0; j < n; j++) sum += ${in0}[j];\n        ${out}[0] = sum;`;
        case 'FILL': return `${out}[i] = 0.0f; // Constant fill`;
        case 'PERMUTE':
        case 'RESHAPE':
        case 'EXPAND':
        case 'SLICE':
        case 'STRIDE':
          return `// View: ${type} - reinterpret memory layout (zero-copy)`;
        default: return `// ${type}: Custom kernel`;
      }
  }
}

// Helper: Generate full kernel code
function generateKernelCode(kernel, backend) {
  const isFused = kernel.isFused || (kernel.ops && kernel.ops.length > 0);
  const name = kernel.name;
  const inputs = kernel.inputs || [];
  const output = kernel.output || 'out';

  let code = '';

  // Header & Signature
  if (backend === 'cuda') {
    // For CUDA, we'll generate a signature with individual pointers for clarity
    const args = [...inputs.map(n => `float* ${n}`), `float* ${output}`, 'int n'].join(', ');
    code += `__global__ void ${name}(${args}) {\n`;
    code += `    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n`;
    code += `    if (idx >= n) return;\n\n`;
  } else if (backend === 'metal') {
    code += `kernel void ${name}(\n`;
    inputs.forEach((inp, i) => {
      code += `    device const float* ${inp} [[buffer(${i})]],\n`;
    });
    code += `    device float* ${output} [[buffer(${inputs.length})]],\n`;
    code += `    uint id [[thread_position_in_grid]]) {\n`;
    code += `    if (id >= n) return;\n\n`;
  } else if (backend === 'opencl') {
    const args = [...inputs.map(n => `__global const float* ${n}`), `__global float* ${output}`, 'int n'].join(', ');
    code += `__kernel void ${name}(${args}) {\n`;
    code += `    int idx = get_global_id(0);\n    if (idx >= n) return;\n\n`;
  } else if (backend === 'wgsl') {
    code += `@compute @workgroup_size(64)\nfn ${name}(@builtin(global_invocation_id) id: vec3<u32>) {\n`;
    code += `    let idx = id.x;\n    if (idx >= n) { return; }\n\n`;
    // WGSL requires bindings, assuming they match order
  } else {
    // C / CPU
    code += `void ${name}(float** inputs_ptr, float** outputs_ptr, int n) {\n`;
    // Unpack inputs/outputs for readability
    inputs.forEach((inp, i) => {
      code += `    float* ${inp} = inputs_ptr[${i}];\n`;
    });
    code += `    float* ${output} = outputs_ptr[0];\n\n`;

    if (backend === 'cpu') {
      code += `    #pragma omp parallel for\n`;
    }
    code += `    for (int i = 0; i < n; i++) {\n`;
  }

  // Body
  if (isFused && kernel.ops) {
    code += `        // Fused Kernel Body\n`;
    kernel.ops.forEach((op, idx) => {
      // Map inputs: check if input is global or local intermediate
      const mappedInputs = op.inputs.map(inp => {
        const isGlobal = inputs.includes(inp);
        // For C/CPU, we access arrays with [i] or [idx]
        // For GPU, we access with [idx] (or [id] for Metal)
        const indexer = (backend === 'c' || backend === 'cpu') ? '[i]' : (backend === 'metal' ? '[id]' : '[idx]');

        if (isGlobal) {
          return `${inp}${indexer}`;
        }
        // Intermediate variable (scalar)
        return inp;
      });

      // Map output
      const isGlobalOutput = (op.output === output);
      const indexer = (backend === 'c' || backend === 'cpu') ? '[i]' : (backend === 'metal' ? '[id]' : '[idx]');
      const mappedOutput = isGlobalOutput ? `${op.output}${indexer}` : op.output;

      const mappedOp = { ...op, inputs: mappedInputs, output: mappedOutput };

      // Indentation
      const indent = (backend === 'c' || backend === 'cpu') ? '        ' : '    ';

      let line = transpileOp(mappedOp, backend, idx);
      // Fix declaration for intermediates
      if (!isGlobalOutput && !line.startsWith('float ')) {
        // It's an assignment to a temp var. In C we need to declare it type if it's new.
        // But we might be reusing t0.
        // Simple heuristic: if it's not global output, prefix with float
        // But only if it's a simple assignment "t0 = ..."
        if (line.match(/^\w+\s*=/)) {
          line = `float ${line}`;
        }
      }

      code += `${indent}${line}\n`;
    });
  } else {
    // Single Op
    const indexer = (backend === 'c' || backend === 'cpu') ? '[i]' : (backend === 'metal' ? '[id]' : '[idx]');
    const mappedInputs = inputs.map(inp => `${inp}${indexer}`);
    const mappedOutput = `${output}${indexer}`;

    const op = { type: kernel.type, inputs: mappedInputs, output: mappedOutput };
    const indent = (backend === 'c' || backend === 'cpu') ? '        ' : '    ';
    code += `${indent}${transpileOp(op, backend)}\n`;
  }

  // Footer
  if (backend === 'c' || backend === 'cpu') {
    code += `    }\n`; // Close for loop
  }
  code += `}\n`;
  return code;
}

const CodeGenView = memo(function CodeGenView({ data }) {
  const [activeAccelerator, setActiveAccelerator] = useState('cuda');

  // Check if we have actual kernel data
  const hasKernelData = data &&
    ((data.unoptimized?.kernels?.length > 0) || (data.optimized?.kernels?.length > 0));

  // Mock kernel data fallback
  const mockKernelData = {
    unoptimized: { kernels: [], deadNodes: 0 },
    optimized: { kernels: [], fusedKernels: 0 }
  };

  const kernelData = data || mockKernelData;

  // Generate full source code for "Original" (Unoptimized)
  const originalSource = useMemo(() => {
    if (!kernelData.unoptimized.kernels.length) return '// No original kernels available';
    return kernelData.unoptimized.kernels.map(k => {
      let header = `// Kernel: ${k.name} (${k.type})`;
      if (k.isDead) header += ' [DEAD CODE - REMOVED]';
      return `${header}\n${generateKernelCode(k, activeAccelerator)}`;
    }).join('\n\n');
  }, [kernelData, activeAccelerator]);

  // Generate full source code for "Optimized"
  const optimizedSource = useMemo(() => {
    if (!kernelData.optimized.kernels.length) return '// No optimized kernels available';
    return kernelData.optimized.kernels.map(k => {
      let header = `// Kernel: ${k.name} (${k.type})`;
      if (k.isFused) header += ' [FUSED KERNEL]';
      return `${header}\n${generateKernelCode(k, activeAccelerator)}`;
    }).join('\n\n');
  }, [kernelData, activeAccelerator]);

  const accelerators = [
    { id: 'c', name: 'C (Scalar)', icon: <Code2 size={16} />, description: 'Portable C99 implementation' },
    { id: 'c_simd', name: 'C (SIMD)', icon: <Zap size={16} />, description: 'AVX2/NEON vector intrinsics' },
    { id: 'cuda', name: 'NVIDIA CUDA', icon: <Server size={16} />, description: 'High-performance GPU backend' },
    { id: 'metal', name: 'Apple Metal', icon: <Smartphone size={16} />, description: 'Optimized for Apple Silicon' },
    { id: 'opencl', name: 'OpenCL', icon: <Monitor size={16} />, description: 'Cross-platform GPU acceleration' },
    { id: 'wgsl', name: 'WebGPU (WGSL)', icon: <Box size={16} />, description: 'Next-gen web graphics' },
    { id: 'cpu', name: 'OpenMP CPU', icon: <Cpu size={16} />, description: 'Multi-threaded CPU fallback' },
  ];

  // Empty state when no kernel data
  if (!hasKernelData) {
    return (
      <div style={{
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-dark)'
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 20,
          padding: 40,
          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(99, 102, 241, 0.02))',
          borderRadius: 16,
          border: '1px solid rgba(99, 102, 241, 0.15)',
          maxWidth: 480
        }}>
          {/* Icon */}
          <div style={{
            width: 80,
            height: 80,
            borderRadius: '50%',
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(99, 102, 241, 0.05))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px solid rgba(99, 102, 241, 0.2)'
          }}>
            <Code2 size={40} color="#6366f1" strokeWidth={1.5} />
          </div>

          {/* Title */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: 'var(--text-primary)', fontSize: 18, fontWeight: 600, marginBottom: 8 }}>
              No Kernel Data
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.5 }}>
              Kernel Studio generates optimized code for multiple accelerator backends.
            </div>
          </div>

          {/* Supported backends */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: 10,
            width: '100%',
            marginTop: 8
          }}>
            {[
              { icon: <Server size={14} />, label: 'CUDA' },
              { icon: <Smartphone size={14} />, label: 'Metal' },
              { icon: <Monitor size={14} />, label: 'OpenCL' },
              { icon: <Cpu size={14} />, label: 'CPU' },
              { icon: <Zap size={14} />, label: 'SIMD' },
              { icon: <Box size={14} />, label: 'WebGPU' }
            ].map((item, i) => (
              <div key={i} style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6,
                padding: '8px 10px',
                background: 'rgba(0, 0, 0, 0.2)',
                borderRadius: 6,
                fontSize: 11,
                color: 'var(--text-secondary)'
              }}>
                {item.icon}
                <span>{item.label}</span>
              </div>
            ))}
          </div>

          {/* Features */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            width: '100%',
            padding: '12px 16px',
            background: 'rgba(0, 0, 0, 0.2)',
            borderRadius: 8
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
              <Trash2 size={14} color="#ef4444" />
              <span>Dead code elimination</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
              <Layers size={14} color="#10b981" />
              <span>Kernel fusion optimization</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
              <Activity size={14} color="#f59e0b" />
              <span>Side-by-side comparison</span>
            </div>
          </div>

          {/* Hint */}
          <div style={{
            marginTop: 4,
            padding: '12px 16px',
            background: 'rgba(16, 185, 129, 0.1)',
            borderRadius: 8,
            border: '1px solid rgba(16, 185, 129, 0.2)',
            display: 'flex',
            alignItems: 'center',
            gap: 10
          }}>
            <Terminal size={16} color="#10b981" />
            <code style={{
              fontSize: 12,
              color: '#10b981',
              fontFamily: 'monospace'
            }}>
              VIZ=1 ./build/bin/dead_code_example
            </code>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      background: 'var(--bg-dark)',
      overflow: 'hidden'
    }}>
      {/* Sidebar: Accelerators */}
      <div style={{
        width: 260,
        background: 'var(--bg-panel)',
        borderRight: '1px solid var(--border-color)',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <div style={{ padding: 20, borderBottom: '1px solid var(--border-color)' }}>
          <h2 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Target Accelerators
          </h2>
        </div>
        <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {accelerators.map(acc => (
            <button
              key={acc.id}
              onClick={() => setActiveAccelerator(acc.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                padding: '12px',
                borderRadius: 8,
                border: activeAccelerator === acc.id ? '1px solid var(--accent-primary)' : '1px solid transparent',
                background: activeAccelerator === acc.id ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.2s'
              }}
            >
              <div style={{
                color: activeAccelerator === acc.id ? 'var(--accent-primary)' : 'var(--text-secondary)',
                display: 'flex'
              }}>
                {acc.icon}
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: activeAccelerator === acc.id ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                  {acc.name}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>
                  {acc.description}
                </div>
              </div>
            </button>
          ))}
        </div>

        <div style={{ marginTop: 'auto', padding: 20, borderTop: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginBottom: 8 }}>
            OPTIMIZATION STATS
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Dead Code</span>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-error)' }}>
              {kernelData.unoptimized.deadNodes} nodes
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Fused Kernels</span>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-success)' }}>
              {kernelData.optimized.fusedKernels} kernels
            </span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Toolbar */}
        <div style={{
          height: 56,
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 24px',
          background: 'rgba(24, 24, 27, 0.5)',
          backdropFilter: 'blur(12px)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <FileCode size={16} color="var(--accent-primary)" />
            <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
              generated_module_{activeAccelerator}.{activeAccelerator === 'cuda' ? 'cu' : activeAccelerator === 'metal' ? 'metal' : activeAccelerator === 'wgsl' ? 'wgsl' : activeAccelerator === 'opencl' ? 'cl' : 'c'}
            </span>
          </div>
        </div>

        {/* Code Editors */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* Original Code Pane */}
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            borderRight: '1px solid var(--border-color)',
            background: '#1e1e1e'
          }}>
            <div style={{
              padding: '8px 16px',
              background: '#252526',
              borderBottom: '1px solid #333',
              fontSize: 12,
              fontWeight: 600,
              color: '#a1a1aa',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span>ORIGINAL IR (Unoptimized)</span>
              <span style={{ color: 'var(--accent-error)', fontSize: 10 }}>
                {kernelData.unoptimized.deadNodes} DEAD NODES
              </span>
            </div>
            <div style={{ flex: 1, overflow: 'auto' }}>
              <SyntaxHighlighter
                language="c"
                style={vscDarkPlus}
                customStyle={{ margin: 0, padding: 24, fontSize: 13, lineHeight: 1.5, background: 'transparent' }}
                showLineNumbers={true}
                wrapLines={true}

              >
                {originalSource}
              </SyntaxHighlighter>
            </div>
          </div>

          {/* Optimized Code Pane */}
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            background: '#1e1e1e'
          }}>
            <div style={{
              padding: '8px 16px',
              background: '#252526',
              borderBottom: '1px solid #333',
              fontSize: 12,
              fontWeight: 600,
              color: '#a1a1aa',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span>OPTIMIZED KERNELS (Fused)</span>
              <span style={{ color: 'var(--accent-success)', fontSize: 10 }}>
                {kernelData.optimized.fusedKernels} FUSED KERNELS
              </span>
            </div>
            <div style={{ flex: 1, overflow: 'auto' }}>
              <SyntaxHighlighter
                language="c"
                style={vscDarkPlus}
                customStyle={{ margin: 0, padding: 24, fontSize: 13, lineHeight: 1.5, background: 'transparent' }}
                showLineNumbers={true}
              >
                {optimizedSource}
              </SyntaxHighlighter>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

export default CodeGenView;
