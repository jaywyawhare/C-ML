import React, { useEffect, useState, useCallback, useMemo, lazy, Suspense, memo, useRef } from 'react';
import { GitBranch, TrendingUp, Code, Activity } from 'lucide-react';

// Lazy load heavy components
const GraphView = lazy(() => import('./components/GraphView.jsx'));
const TrainingEvalView = lazy(() => import('./components/TrainingEvalView.jsx'));
const CodeGenView = lazy(() => import('./components/CodeGenView.jsx'));

// Loading fallback
const LoadingFallback = memo(() => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    color: 'var(--text-secondary)',
    fontSize: 14,
  }}>
    Loading...
  </div>
));

// Custom hook for data fetching with SSE streaming and caching
function useDataFetch(endpoint, enabled = true, useStream = false) {
  const [state, setState] = useState({ data: null, error: null, loading: false });
  const eventSourceRef = useRef(null);

  const fetchData = useCallback(async () => {
    if (!enabled) return;

    setState(s => ({ ...s, loading: true }));
    try {
      const resp = await fetch(endpoint, { cache: 'no-store' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      setState({ data: json.error ? null : json, error: json.error || null, loading: false });
    } catch (e) {
      setState({ data: null, error: e.message, loading: false });
    }
  }, [endpoint, enabled]);

  // SSE streaming for real-time updates
  useEffect(() => {
    if (!enabled || !useStream) {
      fetchData();
      return;
    }

    const streamEndpoint = `${endpoint}/stream`;
    const eventSource = new EventSource(streamEndpoint);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const json = JSON.parse(event.data);
        setState({ data: json.error ? null : json, error: json.error || null, loading: false });
      } catch (e) {
        console.error('Failed to parse SSE data:', e);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      // Fallback to regular fetch
      fetchData();
    };

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [endpoint, enabled, useStream, fetchData]);

  return state;
}

// Tab button component (memoized)
const TabButton = memo(({ id, label, icon: Icon, active, onClick }) => (
  <button
    className={`topbar-button ${active ? 'active' : ''}`}
    onClick={() => onClick(id)}
  >
    <Icon size={16} />
    <span>{label}</span>
  </button>
));

// Tabs configuration
const TABS = [
  { id: 'graph', label: 'Computational Blueprint', icon: GitBranch },
  { id: 'training', label: 'Training', icon: TrendingUp },
  { id: 'codegen', label: 'Kernel Studio', icon: Code },
];

function App() {
  const [activeTab, setActiveTab] = useState('graph');

  // Only fetch data needed for current tab
  const needsGraph = activeTab === 'graph';
  const needsTraining = activeTab === 'graph' || activeTab === 'training';
  const needsKernels = activeTab === 'graph' || activeTab === 'codegen';

  const { data: graph } = useDataFetch('/graph', needsGraph, true);
  const { data: training } = useDataFetch('/training', needsTraining, true);
  const { data: modelArch } = useDataFetch('/model_architecture', needsGraph, false);
  const { data: kernels } = useDataFetch('/kernels', needsKernels, true);

  // Memoize tab click handler
  const handleTabClick = useCallback((id) => setActiveTab(id), []);

  // Memoize view props
  const graphProps = useMemo(() => ({
    graph, training, modelArch, kernels
  }), [graph, training, modelArch, kernels]);

  const trainingProps = useMemo(() => ({
    data: training, mode: 'training'
  }), [training]);

  const codegenProps = useMemo(() => ({
    data: kernels
  }), [kernels]);

  return (
    <div className="app">
      <div className="main-content">
        <div className="topbar">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 32,
              height: 32,
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Activity size={18} color="var(--text-primary)" />
            </div>
            <div className="title">C-ML Visualizer</div>
          </div>

          <div className="topbar-buttons">
            {TABS.map(tab => (
              <TabButton
                key={tab.id}
                {...tab}
                active={activeTab === tab.id}
                onClick={handleTabClick}
              />
            ))}
          </div>

          <div style={{ width: 140 }} />
        </div>

        <div className="panel">
          <div className="panel-body">
            <Suspense fallback={<LoadingFallback />}>
              {activeTab === 'graph' && <GraphView {...graphProps} />}
              {activeTab === 'training' && <TrainingEvalView {...trainingProps} />}
              {activeTab === 'codegen' && <CodeGenView {...codegenProps} />}
            </Suspense>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
