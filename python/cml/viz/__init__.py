"""
C-ML Visualization Module

Launch an interactive dashboard to visualize:
- Computational graphs (Ops Topology)
- Training metrics (Loss, Accuracy curves)
- Kernel Studio (Generated code for different backends)

Usage:
    from cml.viz import launch
    launch()  # Opens browser to http://localhost:8001
"""

from .server import launch, main

__all__ = ["launch", "main"]
