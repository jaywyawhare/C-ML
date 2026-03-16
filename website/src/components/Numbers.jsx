import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

export default function Numbers() {
  const ref = useRef(null)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      gsap.fromTo(ref.current.children,
        { y: 20, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.5, stagger: 0.08, ease: 'power2.out',
          scrollTrigger: { trigger: ref.current, start: 'top 80%' } }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <div id="numbers" ref={ref} className="numbers">
      <div className="num-item" style={{ opacity: 0 }}>
        <div className="num-val">28</div>
        <div className="num-label">NN Layers</div>
      </div>
      <div className="num-item" style={{ opacity: 0 }}>
        <div className="num-val">9</div>
        <div className="num-label">Optimizers</div>
      </div>
      <div className="num-item" style={{ opacity: 0 }}>
        <div className="num-val">7</div>
        <div className="num-label">GPU Backends</div>
      </div>
      <div className="num-item" style={{ opacity: 0 }}>
        <div className="num-val">9</div>
        <div className="num-label">Datasets</div>
      </div>
      <div className="num-item" style={{ opacity: 0 }}>
        <div className="num-val">0</div>
        <div className="num-label">Dependencies</div>
      </div>
    </div>
  )
}
