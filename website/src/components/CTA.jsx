import { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

export default function CTA() {
  const ref = useRef(null)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      gsap.fromTo(ref.current.children,
        { y: 20, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.5, stagger: 0.08, ease: 'power2.out',
          scrollTrigger: { trigger: ref.current, start: 'top 85%' } }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <div ref={ref} className="cta">
      <h2 style={{ opacity: 0 }}>Start training.</h2>
      <p style={{ opacity: 0 }}>Pure C. Zero dependencies. Maximum control.</p>
      <div className="cta-btns" style={{ opacity: 0 }}>
        <Link className="btn primary" to="/docs/getting_started">
          Get Started
        </Link>
        <a className="btn" href="https://github.com/jaywyawhare/C-ML" target="_blank" rel="noopener">
          View on GitHub
        </a>
      </div>
    </div>
  )
}
