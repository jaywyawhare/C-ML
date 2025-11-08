import React from 'react';

export default function CodeGenView({ data }) {
  return (
    <div
      style={{
        position: 'relative',
        height: '100%',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 32,
        background:
          'radial-gradient(circle at 20% 20%, rgba(99,102,241,0.15), transparent 55%), radial-gradient(circle at 80% 30%, rgba(16,185,129,0.12), transparent 50%), radial-gradient(circle at 50% 80%, rgba(239,68,68,0.1), transparent 50%)',
      }}
    >
      <div
        style={{
          position: 'absolute',
          inset: 32,
          borderRadius: 20,
          border: '1px solid rgba(255,255,255,0.08)',
          backdropFilter: 'blur(14px)',
          background: 'linear-gradient(135deg, rgba(15,23,42,0.65), rgba(30,41,59,0.55))',
          boxShadow: '0 25px 60px rgba(15, 23, 42, 0.35)',
        }}
      ></div>

      <div
        style={{
          position: 'relative',
          zIndex: 2,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 18,
          textAlign: 'center',
          color: 'var(--text)',
          maxWidth: 420,
        }}
      >
        <span
          style={{
            fontSize: 12,
            letterSpacing: 6,
            textTransform: 'uppercase',
            color: 'rgba(226,232,240,0.6)',
          }}
        >
          Kernel Studio
        </span>
        <h2
          style={{
            margin: 0,
            fontSize: 32,
            fontWeight: 700,
            background: 'linear-gradient(120deg, #818cf8 0%, #22d3ee 50%, #f472b6 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 8px 24px rgba(129, 140, 248, 0.35)',
          }}
        >
          Coming soon
        </h2>
        <p
          style={{
            margin: 0,
            fontSize: 14,
            lineHeight: 1.6,
            color: 'rgba(226,232,240,0.75)',
          }}
        >
          We&apos;re building an interactive workspace that can emit optimized kernels for CPU, CUDA, Metal and more directly from your C-ML graphs.
        </p>

        <div
          style={{
            display: 'flex',
            gap: 10,
            flexWrap: 'wrap',
            justifyContent: 'center',
          }}
        >
          {['Static analysis', 'Kernel fusion', 'Scheduler insights'].map((chip) => (
            <div
              key={chip}
              style={{
                padding: '6px 14px',
                borderRadius: 999,
                fontSize: 11,
                letterSpacing: 0.3,
                textTransform: 'uppercase',
                color: 'rgba(226,232,240,0.7)',
                background: 'rgba(148, 163, 184, 0.12)',
                border: '1px solid rgba(148, 163, 184, 0.18)',
              }}
            >
              {chip}
            </div>
          ))}
        </div>

        <div className="hanging-slate">
          <div className="hanging-slate__title">Stay tuned for</div>
          <div className="hanging-slate__bullet">
            <span className="hanging-slate__dot dot-indigo"></span>
            <span>Operator scheduling previews</span>
          </div>
          <div className="hanging-slate__bullet">
            <span className="hanging-slate__dot dot-cyan"></span>
            <span>Backend-specific optimizations</span>
          </div>
          <div className="hanging-slate__bullet">
            <span className="hanging-slate__dot dot-rose"></span>
            <span>One-click deployment bundles</span>
          </div>
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background:
            'repeating-linear-gradient(120deg, rgba(148,163,184,0.05) 0px, rgba(148,163,184,0.05) 2px, transparent 2px, transparent 18px)',
        }}
      ></div>
    </div>
  );
}
