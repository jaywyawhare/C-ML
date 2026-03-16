import { useEffect, useRef, useState, useCallback } from 'react'
import gsap from 'gsap'
import { useLossLandscape } from '../hooks/useLossLandscape'

export default function Hero() {
  const canvasRef = useLossLandscape()
  const inner = useRef(null)
  const [copied, setCopied] = useState(false)

  const copy = useCallback(() => {
    navigator.clipboard.writeText('git clone https://github.com/jaywyawhare/C-ML.git')
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }, [])

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ delay: 0.3, defaults: { ease: 'power3.out' } })
      tl.fromTo('.hero-badge', { y: 16, opacity: 0 }, { y: 0, opacity: 1, duration: 0.5 })
        .fromTo('.hero-h1', { y: 24, opacity: 0 }, { y: 0, opacity: 1, duration: 0.6 }, '-=0.3')
        .fromTo('.hero-desc', { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.5 }, '-=0.35')
        .fromTo('.hero-actions', { y: 16, opacity: 0 }, { y: 0, opacity: 1, duration: 0.5 }, '-=0.3')
        .fromTo('.hero-install', { y: 12, opacity: 0 }, { y: 0, opacity: 1, duration: 0.4 }, '-=0.25')
    }, inner)
    return () => ctx.revert()
  }, [])

  return (
    <section className="hero">
      <canvas ref={canvasRef} className="hero-canvas" />
      <div className="hero-fade" />

      <div ref={inner} className="hero-inner">
        <div className="hero-badge" style={{ opacity: 0 }}>
          <span className="hero-dot" />
          v0.0.3 shipped
        </div>

        <h1 className="hero-h1" style={{ opacity: 0 }}>
          Machine learning,<br />
          <em>at the speed of C.</em>
        </h1>

        <p className="hero-desc" style={{ opacity: 0 }}>
          Autograd, neural networks, optimizers, and datasets.
          Pure C core, GPU backends, Python bindings. Zero dependencies.
        </p>

        <div className="hero-actions" style={{ opacity: 0 }}>
          <a className="btn primary" href="#code">Get started</a>
          <a className="btn" href="https://github.com/jaywyawhare/C-ML" target="_blank" rel="noopener">
            View on GitHub
          </a>
        </div>

        <button className="hero-install" style={{ opacity: 0 }} onClick={copy}>
          {copied ? 'copied!' : '$ git clone https://github.com/jaywyawhare/C-ML.git'}
        </button>
      </div>
    </section>
  )
}
