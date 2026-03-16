#!/usr/bin/env python3
"""
C-ML Visualization Server (Zero Dependencies)

Pure Python stdlib HTTP server with SSE streaming, ETag caching, and GZip compression.
No FastAPI, no uvicorn — just http.server + socketserver.
"""

import gzip
import hashlib
import json
import mimetypes
import os
import socketserver
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Configuration
ROOT = Path(__file__).resolve().parent
CWD = Path.cwd()

DATA_FILES = {
    "graph": "graph.json",
    "training": "training.json",
    "model_architecture": "model_architecture.json",
    "kernels": "kernels.json",
}

# In-memory cache: {path_str: (mtime, raw_bytes, parsed_json, etag)}
_cache: dict[str, tuple[float, bytes, dict, str]] = {}
_cache_lock = threading.Lock()


def get_data_path(name: str) -> Path:
    """Get data file path, checking .cml/ first."""
    filename = DATA_FILES.get(name, f"{name}.json")
    cml_path = CWD / ".cml" / filename
    return cml_path if cml_path.exists() else CWD / filename


def load_cached(path: Path) -> tuple:
    """Load JSON with mtime-based caching. Returns (raw_bytes, parsed, etag) or (None, None, None)."""
    if not path.exists():
        return None, None, None

    try:
        key = str(path)
        mtime = path.stat().st_mtime

        with _cache_lock:
            if key in _cache and _cache[key][0] == mtime:
                return _cache[key][1], _cache[key][2], _cache[key][3]

        raw = path.read_bytes()
        parsed = json.loads(raw)
        etag = hashlib.md5(raw).hexdigest()[:12]

        with _cache_lock:
            _cache[key] = (mtime, raw, parsed, etag)

        return raw, parsed, etag
    except Exception:
        return None, None, None


class VizHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with SSE, ETag, GZip, and CORS support."""

    # Suppress default logging
    def log_message(self, format, *args):
        pass

    def end_headers(self):
        # CORS headers on every response
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]  # Strip query params

        # Route dispatch
        if path == "/health":
            self._json_response({"status": "ok"})
        elif path == "/status":
            self._handle_status()
        elif path == "/graph":
            self._handle_data("graph")
        elif path == "/training":
            self._handle_data("training")
        elif path == "/model_architecture":
            self._handle_data("model_architecture")
        elif path == "/kernels":
            self._handle_data("kernels")
        elif path == "/graph/stream":
            self._handle_sse("graph")
        elif path == "/training/stream":
            self._handle_sse("training")
        elif path == "/kernels/stream":
            self._handle_sse("kernels")
        elif path == "/ctxs":
            self._json_response([
                {"name": "Graph", "steps": [{"name": "Graph", "query": "/graph/stream", "depth": 0}]},
                {"name": "Training", "steps": [{"name": "Metrics", "query": "/training/stream", "depth": 0}]},
                {"name": "Kernels", "steps": [{"name": "Kernels", "query": "/kernels/stream", "depth": 0}]},
            ])
        elif path == "/" or path == "/index.html":
            self._serve_static("index.html")
        else:
            # Static file serving from viz/ directory
            rel = path.lstrip("/")
            self._serve_static(rel)

    def _handle_status(self):
        files = {}
        for name in DATA_FILES:
            p = get_data_path(name)
            files[name] = {"exists": p.exists(), "path": str(p)} if p.exists() else {"exists": False}
        self._json_response({
            "version": "0.4.0",
            "cwd": str(CWD),
            "files": files,
            "frontend": True,
        })

    def _handle_data(self, name: str):
        data_path = get_data_path(name)
        raw, parsed, etag = load_cached(data_path)

        if raw is None:
            self._json_response({"error": f"{name} not found"}, status=404)
            return

        # ETag validation
        if_none_match = self.headers.get("If-None-Match")
        if if_none_match and if_none_match == etag:
            self.send_response(304)
            self.end_headers()
            return

        self._send_bytes(raw, "application/json", etag=etag)

    def _handle_sse(self, name: str):
        """Server-Sent Events stream with adaptive polling."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last_mtime = 0.0
        last_data_hash = None
        poll_interval = 0.5
        consecutive_no_change = 0

        # Initial send
        data_path = get_data_path(name)
        if data_path.exists():
            last_mtime = data_path.stat().st_mtime
            raw, parsed, _ = load_cached(data_path)
            if raw:
                last_data_hash = hashlib.md5(raw).hexdigest()
                self._sse_write(raw)
        else:
            self._sse_write(json.dumps({"error": f"{name} not found", "waiting": True}).encode())

        # Watch loop
        try:
            while True:
                time.sleep(poll_interval)
                data_path = get_data_path(name)

                if data_path.exists():
                    mtime = data_path.stat().st_mtime
                    if mtime > last_mtime:
                        last_mtime = mtime
                        raw, parsed, _ = load_cached(data_path)
                        if raw:
                            data_hash = hashlib.md5(raw).hexdigest()
                            if data_hash != last_data_hash:
                                last_data_hash = data_hash
                                consecutive_no_change = 0
                                poll_interval = 0.5
                                self._sse_write(raw)
                            else:
                                consecutive_no_change += 1
                                if consecutive_no_change > 10:
                                    poll_interval = min(2.0, poll_interval * 1.1)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _sse_write(self, data: bytes):
        """Write an SSE data frame. Re-serializes as compact JSON to avoid
        multi-line issues (SSE requires each line prefixed with 'data: ')."""
        try:
            compact = json.dumps(json.loads(data), separators=(",", ":")).encode()
            self.wfile.write(b"data: " + compact + b"\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            raise

    def _json_response(self, obj, status=200):
        raw = json.dumps(obj, separators=(",", ":")).encode()
        self._send_bytes(raw, "application/json", status=status)

    def _send_bytes(self, raw: bytes, content_type: str, status=200, etag=None):
        """Send bytes with optional GZip compression and ETag."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)

        if etag:
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "private, max-age=1")

        # GZip if client supports it and payload is large enough
        accept_encoding = self.headers.get("Accept-Encoding", "")
        if "gzip" in accept_encoding and len(raw) > 100:
            compressed = gzip.compress(raw, compresslevel=6)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(compressed)))
            self.end_headers()
            self.wfile.write(compressed)
        else:
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    def _serve_static(self, rel_path: str):
        """Serve a static file from the viz/ directory."""
        # Security: prevent directory traversal
        safe = Path(rel_path).parts
        if ".." in safe:
            self.send_error(403)
            return

        file_path = ROOT / rel_path
        if not file_path.is_file():
            self.send_error(404)
            return

        # Content type
        ct, _ = mimetypes.guess_type(str(file_path))
        if ct is None:
            ct = "application/octet-stream"

        try:
            raw = file_path.read_bytes()
        except OSError:
            self.send_error(500)
            return

        etag = hashlib.md5(raw).hexdigest()[:12]

        # ETag check
        if_none_match = self.headers.get("If-None-Match")
        if if_none_match and if_none_match == etag:
            self.send_response(304)
            self.end_headers()
            return

        # Cache headers for assets
        cache_control = "public, max-age=31536000, immutable" if "/assets/" in rel_path else "no-cache"

        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", cache_control)

        accept_encoding = self.headers.get("Accept-Encoding", "")
        if "gzip" in accept_encoding and len(raw) > 100:
            compressed = gzip.compress(raw, compresslevel=6)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(compressed)))
            self.end_headers()
            self.wfile.write(compressed)
        else:
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread (needed for SSE)."""
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    host = os.environ.get("HOST", "0.0.0.0")

    server = ThreadedHTTPServer((host, port), VizHandler)
    print(f"C-ML Viz v0.4.0 | http://{host}:{port} | stdlib | threads")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
