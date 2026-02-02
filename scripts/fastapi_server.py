#!/usr/bin/env python3
"""
C-ML Visualization Server (Optimized)

High-performance FastAPI server with gzip compression, file caching, and ETag support.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# Try to use orjson for faster JSON (optional)
try:
    import orjson

    def _dumps(obj) -> bytes:
        return orjson.dumps(obj)

    def _loads(data):
        return orjson.loads(data)

    HAS_ORJSON = True
except ImportError:

    def _dumps(obj) -> bytes:
        return json.dumps(obj, separators=(",", ":")).encode()

    def _loads(data):
        return json.loads(data)

    HAS_ORJSON = False

# Configuration
ROOT = Path(__file__).resolve().parents[1]
CWD = Path.cwd()
VIZ_DIST = ROOT / "viz-ui" / "dist"

DATA_FILES = {
    "graph": "graph.json",
    "training": "training.json",
    "model_architecture": "model_architecture.json",
    "kernels": "kernels.json",
}

# In-memory cache: {path: (mtime, data, etag)}
_cache: dict[str, Tuple[float, dict, str]] = {}


def get_data_path(name: str) -> Path:
    """Get data file path, checking .cml/ first."""
    filename = DATA_FILES.get(name, f"{name}.json")
    cml_path = CWD / ".cml" / filename
    return cml_path if cml_path.exists() else CWD / filename


def load_cached(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    """Load JSON with mtime-based caching. Returns (data, etag)."""
    if not path.exists():
        return None, None

    try:
        key = str(path)
        mtime = path.stat().st_mtime

        # Check cache
        if key in _cache and _cache[key][0] == mtime:
            return _cache[key][1], _cache[key][2]

        # Load fresh
        raw = path.read_bytes()
        data = _loads(raw)
        etag = hashlib.md5(raw).hexdigest()[:12]

        _cache[key] = (mtime, data, etag)
        return data, etag

    except Exception:
        return None, None


class FastJSONResponse(Response):
    """Optimized JSON response."""

    media_type = "application/json"

    def render(self, content) -> bytes:
        return _dumps(content)


# App setup
app = FastAPI(
    title="C-ML Visualization Server",
    version="0.3.0",
    default_response_class=FastJSONResponse,
)

app.add_middleware(
    GZipMiddleware, minimum_size=100
)  # Lower threshold for better compression
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add response headers middleware for caching
class CacheHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Cache static assets longer
        if request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif request.url.path.endswith((".js", ".css", ".png", ".jpg", ".svg")):
            response.headers["Cache-Control"] = "public, max-age=86400"
        return response


app.add_middleware(CacheHeadersMiddleware)


def make_response(
    request: Request, data: Optional[dict], etag: Optional[str], name: str
):
    """Create response with ETag caching."""
    if data is None:
        return FastJSONResponse(
            {"error": f"{name} not found"},
            status_code=status.HTTP_404_NOT_FOUND,
        )

    # ETag validation
    if etag and request.headers.get("if-none-match") == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)

    resp = FastJSONResponse(data)
    if etag:
        resp.headers["ETag"] = etag
        resp.headers["Cache-Control"] = "private, max-age=1"
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "orjson": HAS_ORJSON}


@app.get("/status")
def get_status():
    files = {}
    for name in DATA_FILES:
        path = get_data_path(name)
        files[name] = (
            {"exists": path.exists(), "path": str(path)}
            if path.exists()
            else {"exists": False}
        )
    return {
        "version": "0.3.0",
        "cwd": str(CWD),
        "files": files,
        "frontend": VIZ_DIST.exists(),
    }


@app.get("/graph")
def get_graph(request: Request):
    data, etag = load_cached(get_data_path("graph"))
    return make_response(request, data, etag, "Graph")


@app.get("/training")
def get_training(request: Request):
    data, etag = load_cached(get_data_path("training"))
    return make_response(request, data, etag, "Training")


@app.get("/model_architecture")
def get_model_architecture(request: Request):
    data, etag = load_cached(get_data_path("model_architecture"))
    return make_response(request, data, etag, "Model architecture")


@app.get("/kernels")
def get_kernels(request: Request):
    data, etag = load_cached(get_data_path("kernels"))
    return make_response(request, data, etag, "Kernels")


async def sse_stream(name: str) -> AsyncGenerator[bytes, None]:
    """Optimized SSE file watcher with adaptive polling."""
    last_mtime = 0.0
    last_data_hash = None
    poll_interval = 0.5  # Start with 500ms
    consecutive_no_change = 0

    # Initial send
    path = get_data_path(name)
    if path.exists():
        last_mtime = path.stat().st_mtime
        data, _ = load_cached(path)
        if data:
            data_bytes = _dumps(data)
            last_data_hash = hashlib.md5(data_bytes).hexdigest()
            yield b"data: " + data_bytes + b"\n\n"
    else:
        yield b"data: " + _dumps(
            {"error": f"{name} not found", "waiting": True}
        ) + b"\n\n"

    # Watch loop with adaptive polling
    while True:
        try:
            await asyncio.sleep(poll_interval)
            path = get_data_path(name)

            if path.exists():
                mtime = path.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    data, _ = load_cached(path)
                    if data:
                        data_bytes = _dumps(data)
                        data_hash = hashlib.md5(data_bytes).hexdigest()

                        # Only send if data actually changed
                        if data_hash != last_data_hash:
                            last_data_hash = data_hash
                            consecutive_no_change = 0
                            poll_interval = 0.5  # Reset to fast polling
                            yield b"data: " + data_bytes + b"\n\n"
                        else:
                            consecutive_no_change += 1
                            # Adaptive polling: slow down if no changes
                            if consecutive_no_change > 10:
                                poll_interval = min(2.0, poll_interval * 1.1)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(1)
            poll_interval = 0.5  # Reset on error


def sse_response(gen: AsyncGenerator) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/graph/stream")
async def stream_graph():
    return sse_response(sse_stream("graph"))


@app.get("/training/stream")
async def stream_training():
    return sse_response(sse_stream("training"))


@app.get("/kernels/stream")
async def stream_kernels():
    return sse_response(sse_stream("kernels"))


@app.get("/ctxs")
def get_contexts():
    return [
        {
            "name": "Graph",
            "steps": [{"name": "Graph", "query": "/graph/stream", "depth": 0}],
        },
        {
            "name": "Training",
            "steps": [{"name": "Metrics", "query": "/training/stream", "depth": 0}],
        },
        {
            "name": "Kernels",
            "steps": [{"name": "Kernels", "query": "/kernels/stream", "depth": 0}],
        },
    ]


# Static frontend
if VIZ_DIST.exists():
    app.mount("/", StaticFiles(directory=str(VIZ_DIST), html=True), name="static")
else:

    @app.get("/")
    def root():
        return {
            "message": "C-ML Viz Server",
            "version": "0.3.0",
            "frontend": False,
            "data_dir": str(CWD / ".cml"),
        }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WORKERS", "1"))

    print(
        f"C-ML Viz v0.3.0 | http://{host}:{port} | orjson={'yes' if HAS_ORJSON else 'no'} | workers={workers}"
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        access_log=False,
        log_level="warning",
        workers=workers,
        loop="uvloop" if os.environ.get("USE_UVLOOP", "1") == "1" else "auto",
        limit_concurrency=1000,
        limit_max_requests=10000,
        timeout_keep_alive=30,
    )
