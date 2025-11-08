import React, { useEffect, useState, useRef } from 'react';
import { GitBranch, TrendingUp, Code } from 'lucide-react';
import GraphView from './components/GraphView.jsx';
import TrainingEvalView from './components/TrainingEvalView.jsx';
import CodeGenView from './components/CodeGenView.jsx';

// Use relative paths - Vite will proxy to the HTTP server
const VIZ_SERVER_URL = '';

function App() {
  const [activeTab, setActiveTab] = useState('graph');
  const [graph, setGraph] = useState(null);
  const [training, setTraining] = useState(null);
  const [ts, setTs] = useState(null);
  const [err, setErr] = useState(null);
  const eventSourceRef = useRef(null);
  const trainingEventSourceRef = useRef(null);

  useEffect(() => {
    // Fallback fetch function
    const fetchGraph = async () => {
      try {
        setErr(null);
        const resp = await fetch('/graph', { cache: 'no-store' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const json = await resp.json();
        if (json.error) {
          setErr(json.error);
        } else {
          setGraph(json);
          setTs(new Date());
        }
      } catch (e) {
        setErr('Waiting for graph data...');
      }
    };

    // Use SSE for dynamic updates
    const connectSSE = () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      try {
        setErr(null);
        const eventSource = new EventSource('/graph/stream');
        eventSourceRef.current = eventSource;

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.error) {
              setErr(data.error);
            } else {
              setGraph(data);
              setTs(new Date());
              setErr(null);
            }
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
            setErr('Failed to parse graph data');
          }
        };

        eventSource.onerror = (error) => {
          console.error('SSE connection error:', error);
          setErr('Waiting for graph data...');
          // Try to reconnect after a delay
          setTimeout(() => {
            if (eventSource.readyState === EventSource.CLOSED) {
              connectSSE();
            }
          }, 2000);
        };
      } catch (e) {
        console.error('Failed to create EventSource:', e);
        // Fallback to fetch if SSE fails
        fetchGraph();
      }
    };

    // Connect via SSE when graph tab is active
    if (activeTab === 'graph') {
      // Seed with a one-shot fetch so the panel shows immediately if data exists
      (async () => {
        try {
          const resp = await fetch('/graph', { cache: 'no-store' });
          if (resp.ok) {
            const json = await resp.json();
            if (!json.error) setGraph(json);
          }
        } catch (_) {}
      })();

      // Also fetch training data for Model Architecture view
      (async () => {
        try {
          const resp = await fetch('/training', { cache: 'no-store' });
          if (resp.ok) {
            const json = await resp.json();
            if (!json.error) setTraining(json);
          }
        } catch (_) {}
      })();

      connectSSE();
      // Close training SSE if open
      if (trainingEventSourceRef.current) {
        trainingEventSourceRef.current.close();
        trainingEventSourceRef.current = null;
      }
    } else if (activeTab === 'training') {
      // Seed with a one-shot fetch so the panel shows immediately if data exists
      (async () => {
        try {
          const resp = await fetch('/training', { cache: 'no-store' });
          if (resp.ok) {
            const json = await resp.json();
            if (!json.error) setTraining(json);
          }
        } catch (_) {}
      })();

      // Connect to training SSE
      const connectTrainingSSE = () => {
        if (trainingEventSourceRef.current) {
          trainingEventSourceRef.current.close();
        }

        try {
          setErr(null);
          const eventSource = new EventSource('/training/stream');
          trainingEventSourceRef.current = eventSource;

          eventSource.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              if (data.error) {
                setErr(data.error);
              } else {
                setTraining(data);
                setTs(new Date());
                setErr(null);
              }
            } catch (e) {
              console.error('Failed to parse training SSE data:', e);
              setErr('Failed to parse training data');
            }
          };

          eventSource.onerror = (error) => {
            console.error('Training SSE connection error:', error);
            setErr('Waiting for training data...');
            setTimeout(() => {
              if (eventSource.readyState === EventSource.CLOSED) {
                connectTrainingSSE();
              }
            }, 2000);
          };
        } catch (e) {
          console.error('Failed to create training EventSource:', e);
          // Fallback to fetch
          const fetchTraining = async () => {
            try {
              const resp = await fetch('/training', { cache: 'no-store' });
              if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
              const json = await resp.json();
              if (json.error) {
                setErr(json.error);
              } else {
                setTraining(json);
                setTs(new Date());
              }
            } catch (e) {
              setErr('Waiting for training data...');
            }
          };
          fetchTraining();
        }
      };

      connectTrainingSSE();
      // Close graph SSE if open
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    } else {
      // For other tabs, close SSE connections
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (trainingEventSourceRef.current) {
        trainingEventSourceRef.current.close();
        trainingEventSourceRef.current = null;
      }
    }

    // Reconnect when tab becomes visible
    const onVis = () => {
      if (document.visibilityState === 'visible') {
        if (activeTab === 'graph') {
          connectSSE();
        } else if (activeTab === 'training' && trainingEventSourceRef.current) {
          // Reconnect training SSE if needed
          const connectTrainingSSE = () => {
            if (trainingEventSourceRef.current) {
              trainingEventSourceRef.current.close();
            }
            try {
              setErr(null);
              const eventSource = new EventSource('/training/stream');
              trainingEventSourceRef.current = eventSource;
              eventSource.onmessage = (event) => {
                try {
                  const data = JSON.parse(event.data);
                  if (!data.error) {
                    setTraining(data);
                    setTs(new Date());
                  }
                } catch (e) {
                  console.error('Failed to parse training SSE data:', e);
                }
              };
              eventSource.onerror = () => {
                if (eventSource.readyState === EventSource.CLOSED) {
                  setTimeout(connectTrainingSSE, 2000);
                }
              };
            } catch (e) {
              console.error('Failed to create training EventSource:', e);
            }
          };
          connectTrainingSSE();
        }
      }
    };
    document.addEventListener('visibilitychange', onVis);

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (trainingEventSourceRef.current) {
        trainingEventSourceRef.current.close();
        trainingEventSourceRef.current = null;
      }
      document.removeEventListener('visibilitychange', onVis);
    };
  }, [activeTab]);

  const renderView = () => {
    switch (activeTab) {
      case 'graph':
        return <GraphView graph={graph} training={training} />;
      case 'training':
        return <TrainingEvalView data={training} mode="training" />;
      case 'codegen':
        return <CodeGenView data={null} />;
      default:
        return <GraphView graph={graph} training={training} />;
    }
  };

  const tabs = [
    { id: 'graph', label: 'Computational Blueprint', icon: GitBranch },
    { id: 'training', label: 'Training', icon: TrendingUp },
    { id: 'codegen', label: 'Kernel Studio', icon: Code },
  ];

  return (
    <div className="app">
      <div className="main-content">
        <div className="topbar">
          <div className="title">C-ML Visualizer</div>
          <div className="topbar-buttons">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  className={`topbar-button ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <Icon size={16} className="topbar-button-icon" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
        <div className="panel">
          <div className="panel-body">
            {renderView()}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
