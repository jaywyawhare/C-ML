import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer className="footer">
      <div>
        <div className="footer-brand">C-ML</div>
        <div className="footer-sub">Machine learning, written in C.</div>
      </div>
      <div className="footer-right">
        <a href="https://github.com/jaywyawhare/C-ML" target="_blank" rel="noopener">GitHub</a>
        <a href="#features">Features</a>
        <Link to="/docs/index">Docs</Link>
      </div>
    </footer>
  )
}
