"""Visualization dashboard server."""

import json
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import JSONResponse, FileResponse
    import uvicorn
except ImportError:
    raise ImportError(
        "Visualization dependencies not installed. Install with:\n"
        "  pip install cml[viz]"
    )

PACKAGE_DIR = Path(__file__).parent
STATIC_DIR = PACKAGE_DIR / "static"

app = FastAPI(title="C-ML Visualizer", version="0.1.0")

_working_dir: Path = Path.cwd()


def set_working_dir(path: str | Path):
    """Set the directory the endpoints read their JSON artifacts from."""
    global _working_dir
    _working_dir = Path(path)


def _read_json(filename: str) -> dict:
    """Load ``filename`` from the working dir, returning an ``{"error": ...}`` dict if it is
    missing or malformed rather than raising."""
    filepath = _working_dir / filename
    if filepath.exists():
        try:
            return json.loads(filepath.read_text())
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON in {filename}"}
    return {"error": f"File not found: {filename}"}


@app.get("/graph")
def get_graph():
    """Serve the autograd/IR graph dump (``graph.json``)."""
    return JSONResponse(_read_json("graph.json"))


@app.get("/training")
def get_training():
    """Serve recorded training metrics (``training.json``)."""
    return JSONResponse(_read_json("training.json"))


@app.get("/kernels")
def get_kernels():
    """Serve kernel profiling data (``kernels.json``)."""
    return JSONResponse(_read_json("kernels.json"))


@app.get("/model_architecture")
def get_model_architecture():
    """Serve the model architecture description (``model_architecture.json``)."""
    return JSONResponse(_read_json("model_architecture.json"))


if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:

    @app.get("/")
    def root():
        """Fallback root served when the built static UI is absent; lists the JSON endpoints."""
        return JSONResponse(
            {
                "error": "UI not found",
                "message": "Static UI files not bundled. Run: npm run build in viz-ui/",
                "api_endpoints": [
                    "/graph",
                    "/training",
                    "/kernels",
                    "/model_architecture",
                ],
            }
        )


def launch(
    port: int = 8001,
    host: str = "0.0.0.0",
    open_browser: bool = True,
    working_dir: Optional[str | Path] = None,
    reload: bool = False,
):
    """Start the uvicorn dashboard server, optionally opening a browser tab after a short delay.
    ``reload`` requires the import-string app form, so autoreload re-imports this module."""
    if working_dir:
        set_working_dir(working_dir)

    if open_browser:
        import webbrowser
        import threading

        def _open():
            """Background thread body: wait for the server to bind, then open the browser."""
            import time

            time.sleep(0.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=_open, daemon=True).start()

    print(f"Starting C-ML Visualizer at http://localhost:{port}")
    print(f"Reading JSON from: {_working_dir}")

    uvicorn.run(
        "cml.viz.server:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )


def main():
    """CLI entry point: parse arguments and launch the visualizer server."""
    import argparse

    parser = argparse.ArgumentParser(description="C-ML Visualization Server")
    parser.add_argument(
        "-p", "--port", type=int, default=8001, help="Port (default: 8001)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument(
        "-d", "--dir", type=str, help="Working directory for JSON files"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    launch(
        port=args.port,
        host=args.host,
        open_browser=not args.no_browser,
        working_dir=args.dir,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
