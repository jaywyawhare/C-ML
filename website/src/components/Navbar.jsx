import { useEffect, useRef } from 'react'
import { Link, useLocation } from 'react-router-dom'
import gsap from 'gsap'

export default function Navbar() {
  const ref = useRef(null)
  const { pathname } = useLocation()
  const isHome = pathname === '/'

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      gsap.fromTo(ref.current,
        { y: -20, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, ease: 'power2.out', delay: 0.1 }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <nav ref={ref} className="nav" style={{ opacity: 0 }}>
      <Link className="nav-logo" to="/">
        <img className="nav-logo-icon" src="https://raw.githubusercontent.com/jaywyawhare/C-ML/master/docs/light-mode.svg" alt="C-ML" />
        <span className="nav-ver">0.0.2</span>
      </Link>

      <ul className="nav-links">
        {isHome ? (
          <>
            <li><a href="#features">Features</a></li>
            <li><a href="#code">Code</a></li>
            <li><a href="#architecture">Architecture</a></li>
          </>
        ) : null}
        <li><Link to="/docs/index">Docs</Link></li>
      </ul>

      <div className="nav-right">
        <a className="btn-sm" href="https://github.com/jaywyawhare/C-ML" target="_blank" rel="noopener">GitHub</a>
        <Link className="btn-sm filled" to="/docs/getting_started">Get Started</Link>
      </div>
    </nav>
  )
}
