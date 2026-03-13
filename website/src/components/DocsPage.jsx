import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash'
import c from 'react-syntax-highlighter/dist/esm/languages/prism/c'
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json'
import yaml from 'react-syntax-highlighter/dist/esm/languages/prism/yaml'
import makefile from 'react-syntax-highlighter/dist/esm/languages/prism/makefile'
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python'

SyntaxHighlighter.registerLanguage('bash', bash)
SyntaxHighlighter.registerLanguage('sh', bash)
SyntaxHighlighter.registerLanguage('shell', bash)
SyntaxHighlighter.registerLanguage('c', c)
SyntaxHighlighter.registerLanguage('json', json)
SyntaxHighlighter.registerLanguage('yaml', yaml)
SyntaxHighlighter.registerLanguage('yml', yaml)
SyntaxHighlighter.registerLanguage('makefile', makefile)
SyntaxHighlighter.registerLanguage('python', python)
SyntaxHighlighter.registerLanguage('py', python)

const sections = [
  {
    group: 'Getting Started',
    icon: '\u25B6',
    items: [
      { slug: 'index', label: 'Overview', file: 'index.md' },
      { slug: 'getting_started', label: 'Getting Started', file: 'getting_started.md' },
      { slug: 'training', label: 'Training Guide', file: 'training.md' },
    ],
  },
  {
    group: 'Core',
    icon: '\u2039\u203A',
    items: [
      { slug: 'api_reference', label: 'API Reference', file: 'api_reference.md' },
      { slug: 'nn_layers', label: 'Neural Network Layers', file: 'nn_layers.md' },
      { slug: 'autograd', label: 'Autograd', file: 'autograd.md' },
      { slug: 'datasets', label: 'Datasets', file: 'datasets.md' },
    ],
  },
  {
    group: 'Advanced',
    icon: '\u2302',
    items: [
      { slug: 'graph_mode', label: 'Graph Mode', file: 'graph_mode.md' },
      { slug: 'ir_graph_management', label: 'IR Graph Management', file: 'ir_graph_management.md' },
      { slug: 'optimizations', label: 'Optimizations', file: 'optimizations.md' },
      { slug: 'linearization', label: 'Linearization & Fused Codegen', file: 'linearization.md' },
      { slug: 'beam_search', label: 'BEAM Search Auto-Tuning', file: 'beam_search.md' },
      { slug: 'speculative_decoding', label: 'Speculative Decoding', file: 'speculative_decoding.md' },
    ],
  },
  {
    group: 'Tools',
    icon: '\u2261',
    items: [
      { slug: 'kernel_studio', label: 'Kernel Studio', file: 'kernel_studio.md' },
      { slug: 'kernel_studio_quickref', label: 'Kernel Studio Quick Ref', file: 'kernel_studio_quickref.md' },
      { slug: 'benchmarks', label: 'Benchmarks', file: 'benchmarks.md' },
    ],
  },
  {
    group: 'Reference',
    icon: '\u2764',
    items: [
      { slug: 'EXTERNAL_DEPENDENCIES', label: 'External Dependencies', file: 'EXTERNAL_DEPENDENCIES.md' },
    ],
  },
]

const allItems = sections.flatMap(s => s.items)

function stripFrontmatter(text) {
  if (text.startsWith('---')) {
    const end = text.indexOf('---', 3)
    if (end !== -1) return text.slice(end + 3).trim()
  }
  return text
}

function extractHeadings(md) {
  const headings = []
  const lines = md.split('\n')
  for (const line of lines) {
    const m = line.match(/^(#{2,3})\s+(.+)/)
    if (m) {
      headings.push({
        level: m[1].length,
        text: m[2].replace(/[`*_~]/g, ''),
        id: m[2].replace(/[`*_~]/g, '').toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-'),
      })
    }
  }
  return headings
}

const codeTheme = {
  ...oneDark,
  'pre[class*="language-"]': {
    ...oneDark['pre[class*="language-"]'],
    background: '#15130f',
    margin: 0,
    borderRadius: 0,
    padding: '20px 24px',
    fontSize: '0.78rem',
    lineHeight: 1.7,
  },
  'code[class*="language-"]': {
    ...oneDark['code[class*="language-"]'],
    background: 'none',
    fontSize: '0.78rem',
    lineHeight: 1.7,
  },
}

function CodeBlock({ className, children }) {
  const [copied, setCopied] = useState(false)
  const match = /language-(\w+)/.exec(className || '')
  const lang = match ? match[1] : ''
  const code = String(children).replace(/\n$/, '')

  const onCopy = useCallback(() => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }, [code])

  return (
    <div className="docs-code-block">
      <div className="docs-code-header">
        <span className="docs-code-lang">{lang || 'text'}</span>
        <button className="docs-code-copy" onClick={onCopy}>
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <SyntaxHighlighter
        style={codeTheme}
        language={lang || 'text'}
        PreTag="div"
        wrapLongLines
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
}

function TableOfContents({ headings, activeId }) {
  if (headings.length === 0) return null

  return (
    <aside className="docs-toc">
      <div className="docs-toc-title">On this page</div>
      <nav className="docs-toc-nav">
        {headings.map((h, i) => (
          <a
            key={i}
            href={`#${h.id}`}
            className={`docs-toc-link ${h.level === 3 ? 'docs-toc-sub' : ''} ${activeId === h.id ? 'active' : ''}`}
          >
            {h.text}
          </a>
        ))}
      </nav>
    </aside>
  )
}

function PrevNext({ currentSlug }) {
  const idx = allItems.findIndex(i => i.slug === currentSlug)
  const prev = idx > 0 ? allItems[idx - 1] : null
  const next = idx < allItems.length - 1 ? allItems[idx + 1] : null

  if (!prev && !next) return null

  return (
    <div className="docs-prevnext">
      {prev ? (
        <Link to={`/docs/${prev.slug}`} className="docs-prevnext-link docs-prev">
          <span className="docs-prevnext-dir">Previous</span>
          <span className="docs-prevnext-label">{prev.label}</span>
        </Link>
      ) : <div />}
      {next ? (
        <Link to={`/docs/${next.slug}`} className="docs-prevnext-link docs-next">
          <span className="docs-prevnext-dir">Next</span>
          <span className="docs-prevnext-label">{next.label}</span>
        </Link>
      ) : <div />}
    </div>
  )
}

export default function DocsPage() {
  const { slug } = useParams()
  const navigate = useNavigate()
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(true)
  const [activeId, setActiveId] = useState('')
  const contentRef = useRef(null)

  const active = slug || 'index'

  const headings = useMemo(() => extractHeadings(content), [content])

  const loadDoc = useCallback((docSlug) => {
    const item = allItems.find(i => i.slug === docSlug)
    if (!item) {
      setContent('# Page not found\n\nThe requested documentation page does not exist.')
      setLoading(false)
      return
    }

    setLoading(true)
    fetch(`${import.meta.env.BASE_URL}docs/${item.file}`)
      .then(r => {
        if (!r.ok) throw new Error('Not found')
        return r.text()
      })
      .then(text => {
        setContent(stripFrontmatter(text))
        setLoading(false)
        window.scrollTo(0, 0)
      })
      .catch(() => {
        setContent('# Error\n\nCould not load this documentation page.')
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    if (!slug) {
      navigate('/docs/index', { replace: true })
      return
    }
    loadDoc(active)
  }, [active, slug, navigate, loadDoc])

  // Track active heading for ToC highlight
  useEffect(() => {
    if (headings.length === 0) return

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id)
          }
        }
      },
      { rootMargin: '-80px 0px -70% 0px', threshold: 0 }
    )

    const timer = setTimeout(() => {
      const el = contentRef.current
      if (!el) return
      el.querySelectorAll('h2[id], h3[id]').forEach(h => observer.observe(h))
    }, 200)

    return () => {
      clearTimeout(timer)
      observer.disconnect()
    }
  }, [headings, content])

  // Find current section for breadcrumb
  const currentSection = sections.find(s => s.items.some(i => i.slug === active))
  const currentItem = allItems.find(i => i.slug === active)

  return (
    <div className="docs-layout">
      <aside className="docs-sidebar">
        <div className="docs-sidebar-title">Documentation</div>
        <nav className="docs-nav">
          {sections.map(section => (
            <div key={section.group}>
              <div className="docs-nav-group">
                <span className="docs-nav-group-icon">{section.icon}</span>
                {section.group}
              </div>
              {section.items.map(item => (
                <Link
                  key={item.slug}
                  to={`/docs/${item.slug}`}
                  className={active === item.slug ? 'active' : ''}
                >
                  {item.label}
                </Link>
              ))}
            </div>
          ))}
        </nav>
      </aside>

      <main className="docs-content" ref={contentRef}>
        {/* Breadcrumb */}
        {currentSection && currentItem && (
          <div className="docs-breadcrumb">
            <Link to="/docs/index">Docs</Link>
            <span className="docs-breadcrumb-sep">/</span>
            <span>{currentSection.group}</span>
            <span className="docs-breadcrumb-sep">/</span>
            <span className="docs-breadcrumb-current">{currentItem.label}</span>
          </div>
        )}

        {loading ? (
          <div className="docs-loading">
            <div className="docs-loading-bar" />
            Loading documentation...
          </div>
        ) : (
          <>
            <Markdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeRaw]}
              components={{
                code({ className, children, node, ...props }) {
                  const isInline = !className && !String(children).includes('\n')
                  if (isInline) {
                    return <code className="docs-inline-code" {...props}>{children}</code>
                  }
                  return <CodeBlock className={className}>{children}</CodeBlock>
                },
                h2({ children, ...props }) {
                  const text = String(children).replace(/[`*_~]/g, '')
                  const id = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-')
                  return <h2 id={id} {...props}>{children}</h2>
                },
                h3({ children, ...props }) {
                  const text = String(children).replace(/[`*_~]/g, '')
                  const id = text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-')
                  return <h3 id={id} {...props}>{children}</h3>
                },
                a({ href, children, ...props }) {
                  if (!href) return <a {...props}>{children}</a>
                  // Handle .md links -> SPA routes
                  if (href.endsWith('.md')) {
                    const docSlug = href.replace(/^(\.\.\/)*/, '').replace(/\.md$/, '')
                    if (allItems.some(i => i.slug === docSlug || i.file === href.replace(/^(\.\.\/)*/, ''))) {
                      const found = allItems.find(i => i.slug === docSlug || i.file === href.replace(/^(\.\.\/)*/, ''))
                      if (found) return <Link to={`/docs/${found.slug}`} {...props}>{children}</Link>
                    }
                  }
                  // Handle /docs/slug links
                  if (href.startsWith('/docs/')) {
                    const docSlug = href.replace(/^\/docs\//, '')
                    if (allItems.some(i => i.slug === docSlug)) {
                      return <Link to={href} {...props}>{children}</Link>
                    }
                  }
                  // Anchor links stay as-is, external links open in new tab
                  if (href.startsWith('#')) return <a href={href} {...props}>{children}</a>
                  if (href.startsWith('http')) return <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>
                  return <a href={href} {...props}>{children}</a>
                },
                table({ children, ...props }) {
                  return (
                    <div className="docs-table-wrap">
                      <table {...props}>{children}</table>
                    </div>
                  )
                },
              }}
            >
              {content}
            </Markdown>
            <PrevNext currentSlug={active} />
          </>
        )}
      </main>

      <TableOfContents headings={headings} activeId={activeId} />
    </div>
  )
}
