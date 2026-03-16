import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const features = [
  {
    name: 'Autograd',
    text: 'Dynamic computation graphs with automatic gradient computation. Gradient checkpointing trades compute for memory.',
    tags: ['Backprop', 'Dynamic Graph', 'Checkpointing'],
  },
  {
    name: '28 Neural Network Layers',
    text: 'Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, Transformer, Embedding, BatchNorm, LayerNorm, RMSNorm, InstanceNorm, Pooling, PixelShuffle, Upsample.',
    tags: ['28 Layers', 'Containers'],
  },
  {
    name: 'LLM Support',
    text: 'LoRA/QLoRA fine-tuning, Flash Attention, GQA, Paged Attention, RoPE, MoE, Speculative Decoding, continuous batching serving.',
    tags: ['LoRA', 'LLaMA', 'Serving'],
  },
  {
    name: '7 GPU Backends',
    text: 'CUDA (PTX), ROCm (KFD), Vulkan (SPIR-V), WebGPU (WGSL), Metal (MSL), OpenCL, LLVM JIT. Automatic dispatch with CPU fallback.',
    tags: ['CUDA', 'Vulkan', 'Metal'],
  },
  {
    name: 'Compiler Pipeline',
    text: 'IR optimization, pattern matching, operator fusion, linearization, fused codegen. AOT and JIT compilation with kernel caching.',
    tags: ['Fusion', 'AOT', 'Z3 Verify'],
  },
  {
    name: 'Distributed Training',
    text: 'DDP with bucketed all-reduce, GPipe pipeline parallelism, Megatron tensor parallelism. NCCL/MPI/Gloo backends.',
    tags: ['DDP', 'Pipeline', 'Tensor Parallel'],
  },
  {
    name: 'Model I/O',
    text: 'Load and save GGUF, SafeTensors, ONNX, PyTorch .pth. Quantization: int8, NF4. Architecture export to JSON.',
    tags: ['GGUF', 'ONNX', 'Quantization'],
  },
  {
    name: 'SIMD / BLAS / Memory',
    text: 'SSE/AVX/AVX-512/NEON with runtime detection. Dynamic MKL/OpenBLAS. TLSF allocator, memory pools, graph allocator.',
    tags: ['AVX-512', 'BLAS', 'TLSF'],
  },
  {
    name: 'Optimizers & Losses',
    text: '9 optimizers (Adam, AdamW, SGD, RMSprop, etc.) with 6 LR schedulers. 13 loss functions including Focal, Triplet, CosineEmbedding.',
    tags: ['Adam', '13 Losses', 'Schedulers'],
  },
  {
    name: 'Dataset Hub & Model Zoo',
    text: 'One-liner loading for 9 datasets. Pre-built MLP, ResNet, VGG, GPT-2, BERT, LLaMA (7B/13B/70B) architectures.',
    tags: ['9 Datasets', 'LLaMA', 'BERT'],
  },
  {
    name: 'Python Bindings',
    text: 'CFFI-based Python interface with NumPy integration. Zero-copy tensor sharing, full operator overloading, context managers.',
    tags: ['CFFI', 'NumPy', 'Zero-Copy'],
  },
  {
    name: 'Profiling & Debugging',
    text: 'Built-in timers, memory tracking, Kernel Studio visualization. IR graph export to JSON, ONNX, and DOT formats.',
    tags: ['Kernel Studio', 'Profiler', 'IR Export'],
  },
]

export default function Features() {
  const ref = useRef(null)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const el = ref.current
    if (!el) return

    const ctx = gsap.context(() => {
      gsap.fromTo('.section-header-feat',
        { y: 24, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, ease: 'power2.out',
          scrollTrigger: { trigger: el, start: 'top 85%' } }
      )
      gsap.fromTo(el.querySelectorAll('.feat'),
        { y: 30, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.5, stagger: 0.04, ease: 'power2.out',
          scrollTrigger: { trigger: el, start: 'top 75%' } }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <section id="features" ref={ref}>
      <div className="section-header-feat" style={{ opacity: 0, marginBottom: 48 }}>
        <div className="section-label">Capabilities</div>
        <h2 className="section-heading">Everything you need to train. Nothing you don't.</h2>
        <p className="section-sub">From tensors to transformers, in pure C.</p>
      </div>

      <div className="feat-grid">
        {features.map((f, i) => (
          <div key={i} className="feat" style={{ opacity: 0 }}>
            <div className="feat-name">{f.name}</div>
            <div className="feat-text">{f.text}</div>
            <div className="feat-tags">
              {f.tags.map((t, j) => <span key={j} className="feat-tag">{t}</span>)}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
