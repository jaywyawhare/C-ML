import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Features from './components/Features'
import CodeShowcase from './components/CodeShowcase'
import Architecture from './components/Architecture'
import Numbers from './components/Numbers'
import CTA from './components/CTA'
import Footer from './components/Footer'
import DocsPage from './components/DocsPage'

function Landing() {
  return (
    <>
      <Hero />
      <div className="divider" />
      <Features />
      <div className="divider" />
      <CodeShowcase />
      <div className="divider" />
      <Architecture />
      <div className="divider" />
      <Numbers />
      <CTA />
    </>
  )
}

export default function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="/docs/:slug" element={<DocsPage />} />
      </Routes>
      <Footer />
    </>
  )
}
