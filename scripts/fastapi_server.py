import os
import json
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "viz-ui" / "public"
GRAPH_FILE = PUBLIC / "graph.json"
TRAINING_FILE = PUBLIC / "training.json"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_json(p: Path):
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/graph")
def get_graph():
    data = _load_json(GRAPH_FILE)
    if data is None:
        return JSONResponse(
            {"error": "Graph file not found"}, status_code=status.HTTP_404_NOT_FOUND
        )
    return JSONResponse(data)


@app.get("/training")
def get_training():
    data = _load_json(TRAINING_FILE)
    if data is None:
        return JSONResponse(
            {"error": "Training file not found"}, status_code=status.HTTP_404_NOT_FOUND
        )
    return JSONResponse(data)


async def _sse_file_stream(path: Path) -> AsyncGenerator[bytes, None]:
    last_mtime = path.stat().st_mtime if path.exists() else 0.0
    if path.exists():
        yield b"data: " + json.dumps(_load_json(path)).encode() + b"\n\n"
    while True:
        try:
            await asyncio.sleep(0.5)
            if path.exists():
                m = path.stat().st_mtime
                if m > last_mtime:
                    last_mtime = m
                    yield b"data: " + json.dumps(_load_json(path)).encode() + b"\n\n"
        except asyncio.CancelledError:
            break
        except Exception:
            # send error then continue
            yield b'data: {"error": "stream error"}\n\n'


@app.get("/graph/stream")
async def stream_graph():
    if not GRAPH_FILE.exists():
        return StreamingResponse(
            (chunk for chunk in [b'data: {"error": "Graph file not found"}\n\n']),
            media_type="text/event-stream",
        )
    return StreamingResponse(
        _sse_file_stream(GRAPH_FILE), media_type="text/event-stream"
    )


@app.get("/training/stream")
async def stream_training():
    if not TRAINING_FILE.exists():
        return StreamingResponse(
            (chunk for chunk in [b'data: {"error": "Training file not found"}\n\n']),
            media_type="text/event-stream",
        )
    return StreamingResponse(
        _sse_file_stream(TRAINING_FILE), media_type="text/event-stream"
    )


@app.get("/ctxs")
def get_ctxs():
    return JSONResponse(
        [
            {
                "name": "Gradient Graph",
                "steps": [
                    {"name": "Computation Graph", "query": "/graph/stream", "depth": 0}
                ],
            },
            {
                "name": "Training",
                "steps": [{"name": "Metrics", "query": "/training/stream", "depth": 0}],
            },
        ]
    )


@app.get("/")
def root():
    return JSONResponse({"ok": True, "graph": "/graph", "training": "/training"})


if __name__ == "__main__":
    import asyncio
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    PUBLIC.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)
