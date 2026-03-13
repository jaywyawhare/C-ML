import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const features = [
  {
    name: 'Autograd',
    text: 'Dynamic computation graphs with automatic gradient computation. Full backward pass through arbitrary DAGs.',
    tags: ['Backprop', 'Dynamic Graph', 'Chain Rule'],
  },
  {
    name: 'Neural Network Layers',
    text: 'Linear, Conv1d/2d/3d, RNN/LSTM/GRU, Transformer, Embedding, BatchNorm, LayerNorm, GroupNorm, Pooling, Dropout.',
    tags: ['20+ Layers', 'Containers'],
  },
  {
    name: 'Multi-GPU Backends',
    text: 'CUDA, ROCm, Metal, WebGPU, and LLVM backends with automatic dispatch. Fallback chain to CPU.',
    tags: ['CUDA', 'Metal', 'WebGPU'],
  },
  {
    name: 'Optimizers',
    text: 'SGD, Adam, AdamW, RMSprop, Adagrad, AdaDelta. LR schedulers: Step, Exponential, Cosine, ReduceOnPlateau.',
    tags: ['Adam', 'Cosine LR', 'Schedulers'],
  },
  {
    name: 'Loss Functions',
    text: 'MSE, MAE, BCE, CrossEntropy, Huber, KL Divergence, Hinge, Focal, SmoothL1. Differentiable and composable.',
    tags: ['9 Losses', 'Differentiable'],
  },
  {
    name: 'SIMD / BLAS / JIT',
    text: 'SSE/AVX/AVX-512/NEON with runtime detection. Dynamic MKL/OpenBLAS loading. IR graph optimization and fusion.',
    tags: ['AVX-512', 'NEON', 'IR Fusion'],
  },
  {
    name: 'Dataset Hub',
    text: 'One-liner loading: cml_dataset_load("iris"). Auto-download and caching for MNIST, CIFAR-10, and 7 more.',
    tags: ['9 Datasets', 'Auto-Cache'],
  },
  {
    name: 'Model Zoo',
    text: 'Pre-built architectures: MLP, ResNet, VGG, GPT-2, BERT. Save/load models and training checkpoints.',
    tags: ['ResNet', 'GPT-2', 'BERT'],
  },
  {
    name: 'Python Bindings',
    text: 'CFFI-based Python interface. Use C-ML from Python with zero-copy tensor sharing and full API access.',
    tags: ['CFFI', 'Zero-Copy'],
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
