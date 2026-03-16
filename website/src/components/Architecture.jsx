import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const layers = [
  // Layer 1: User-facing bindings
  {
    label: 'Python CFFI',
    sub: 'NumPy integration, zero-copy bindings',
    tier: 'top',
    cols: 1,
    items: null,
  },
  // Layer 2: Public C API
  {
    label: 'Public API (cml.h)',
    sub: 'Tensor creation, forward pass, backward, training loop',
    tier: 'top',
    cols: 1,
    items: null,
  },
  // Layer 3: High-level ML components
  {
    label: null,
    sub: null,
    tier: 'mid',
    cols: 4,
    items: [
      { label: 'NN Layers', sub: '28 layers, containers' },
      { label: 'LLM Ops', sub: 'LoRA, GQA, RoPE, MoE' },
      { label: 'Optimizers', sub: 'Adam, SGD, 6 schedulers' },
      { label: 'Loss Functions', sub: '13 differentiable losses' },
    ],
  },
  // Layer 4: Training infrastructure
  {
    label: null,
    sub: null,
    tier: 'mid',
    cols: 4,
    items: [
      { label: 'Autograd', sub: 'Dynamic graphs, checkpointing' },
      { label: 'Distributed', sub: 'DDP, pipeline, tensor parallel' },
      { label: 'Serving', sub: 'Batching, KV cache, speculative' },
      { label: 'Model I/O', sub: 'GGUF, ONNX, SafeTensors, .pth' },
    ],
  },
  // Layer 5: Tensor runtime
  {
    label: 'Tensor Operations',
    sub: 'Broadcasting, SIMD (SSE/AVX/AVX-512/NEON), BLAS',
    tier: 'mid',
    cols: 1,
    items: null,
  },
  // Layer 6: Compiler pipeline
  {
    label: null,
    sub: null,
    tier: 'low',
    cols: 4,
    items: [
      { label: 'IR Graph', sub: 'DCE, fusion, Z3 verify' },
      { label: 'Linearizer', sub: 'Virtual register alloc' },
      { label: 'Codegen', sub: 'C, PTX, SPIR-V, WGSL, MSL' },
      { label: 'Kernel Cache', sub: 'LRU, AOT, JIT' },
    ],
  },
  // Layer 7: Device abstraction
  {
    label: null,
    sub: null,
    tier: 'low',
    cols: 4,
    items: [
      { label: 'Memory', sub: 'TLSF, pools, graph alloc' },
      { label: 'HCQ', sub: 'Hardware command queues' },
      { label: 'NV Driver', sub: 'RM ioctl, GPFIFO rings' },
      { label: 'AM Driver', sub: 'KFD ioctl, AQL dispatch' },
    ],
  },
  // Layer 8: Hardware backends
  {
    label: null,
    sub: null,
    tier: 'bottom',
    cols: 7,
    items: [
      { label: 'CUDA' },
      { label: 'ROCm' },
      { label: 'Vulkan' },
      { label: 'Metal' },
      { label: 'WebGPU' },
      { label: 'OpenCL' },
      { label: 'CPU' },
    ],
  },
]

function SingleLayer({ label, sub, tier, style }) {
  return (
    <div className={`arch-card arch-${tier}`} style={style}>
      <div className="arch-card-label">{label}</div>
      {sub && <div className="arch-card-sub">{sub}</div>}
    </div>
  )
}

function MultiLayer({ items, tier, cols, style }) {
  return (
    <div className={`arch-row arch-row-${cols}`} style={style}>
      {items.map((item, j) => (
        <div key={j} className={`arch-card arch-${tier}`}>
          <div className="arch-card-label">{item.label}</div>
          {item.sub && <div className="arch-card-sub">{item.sub}</div>}
        </div>
      ))}
    </div>
  )
}

function Connector({ style }) {
  return (
    <div className="arch-connector" style={style}>
      <svg width="2" height="20" viewBox="0 0 2 20">
        <line x1="1" y1="0" x2="1" y2="20" stroke="rgba(245,158,11,0.2)" strokeWidth="2" strokeDasharray="3,3" />
      </svg>
    </div>
  )
}

export default function Architecture() {
  const ref = useRef(null)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      gsap.fromTo('.arch-header',
        { y: 24, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, ease: 'power2.out',
          scrollTrigger: { trigger: ref.current, start: 'top 85%' } }
      )
      gsap.fromTo(ref.current.querySelectorAll('.arch-card, .arch-row, .arch-connector'),
        { y: 24, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.45, stagger: 0.06, ease: 'power2.out',
          scrollTrigger: { trigger: ref.current, start: 'top 75%' } }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <section id="architecture" ref={ref}>
      <div className="arch-header" style={{ opacity: 0, marginBottom: 48 }}>
        <div className="section-label">Architecture</div>
        <h2 className="section-heading">Clean layers. Zero magic.</h2>
        <p className="section-sub">Every layer maps to a directory in the source tree.</p>
      </div>

      <div className="arch-diagram">
        {layers.map((layer, i) => (
          <div key={i}>
            {i > 0 && <Connector style={{ opacity: 0 }} />}
            {layer.items ? (
              <MultiLayer items={layer.items} tier={layer.tier} cols={layer.cols} style={{ opacity: 0 }} />
            ) : (
              <SingleLayer label={layer.label} sub={layer.sub} tier={layer.tier} style={{ opacity: 0 }} />
            )}
          </div>
        ))}
      </div>
    </section>
  )
}
