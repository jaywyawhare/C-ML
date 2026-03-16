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
    global _working_dir
    _working_dir = Path(path)


def _read_json(filename: str) -> dict:
    filepath = _working_dir / filename
    if filepath.exists():
        try:
            return json.loads(filepath.read_text())
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON in {filename}"}
    return {"error": f"File not found: {filename}"}


@app.get("/graph")
def get_graph():
    return JSONResponse(_read_json("graph.json"))


@app.get("/training")
def get_training():
    return JSONResponse(_read_json("training.json"))


@app.get("/kernels")
def get_kernels():
    return JSONResponse(_read_json("kernels.json"))


@app.get("/model_architecture")
def get_model_architecture():
    return JSONResponse(_read_json("model_architecture.json"))


if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:

    @app.get("/")
    def root():
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
    if working_dir:
        set_working_dir(working_dir)

    if open_browser:
        import webbrowser
        import threading

        def _open():
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
