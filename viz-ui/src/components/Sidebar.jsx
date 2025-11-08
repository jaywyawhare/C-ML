import React from 'react';
import { Network, TrendingUp, Code } from 'lucide-react';

export default function Sidebar({ activeTab, onTabChange }) {
  const tabs = [
    { id: 'graph', label: 'Gradient Graph', icon: Network },
    { id: 'training', label: 'Training & Eval', icon: TrendingUp },
    { id: 'codegen', label: 'Generated Code', icon: Code },
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-title">C-ML Viz</div>
      </div>
      <div className="sidebar-tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              className={`sidebar-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => onTabChange(tab.id)}
            >
              <Icon size={18} className="sidebar-tab-icon" />
              <span className="sidebar-tab-label">{tab.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
