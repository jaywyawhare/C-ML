#!/usr/bin/env bash
# Download vendored JS/CSS libraries for C-ML Visualizer
# Run once: bash viz/fetch_assets.sh

set -euo pipefail
DIR="$(cd "$(dirname "$0")/assets" && pwd)"
mkdir -p "$DIR" "$DIR/languages" "$DIR/styles"

echo "Downloading vendored assets to $DIR ..."

# Cytoscape + layout extensions
curl -sL "https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"        -o "$DIR/cytoscape.min.js"
curl -sL "https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"                  -o "$DIR/dagre.min.js"
curl -sL "https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"       -o "$DIR/cytoscape-dagre.js"
curl -sL "https://unpkg.com/elkjs@0.9.3/lib/elk.bundled.js"                 -o "$DIR/elk.bundled.js"
curl -sL "https://unpkg.com/cytoscape-elk@2.2.0/dist/cytoscape-elk.js"      -o "$DIR/cytoscape-elk.js"

# D3.js v7
curl -sL "https://unpkg.com/d3@7.9.0/dist/d3.min.js"                       -o "$DIR/d3.min.js"

# highlight.js (core + C language + VS2015 dark theme)
curl -sL "https://unpkg.com/@highlightjs/cdn-assets@11.11.1/highlight.min.js"                -o "$DIR/highlight.min.js"
curl -sL "https://unpkg.com/@highlightjs/cdn-assets@11.11.1/languages/c.min.js"              -o "$DIR/languages/c.min.js"
curl -sL "https://unpkg.com/@highlightjs/cdn-assets@11.11.1/styles/vs2015.min.css"           -o "$DIR/styles/vs2015.min.css"

echo "Done. $(find "$DIR" -type f | wc -l) files downloaded."
