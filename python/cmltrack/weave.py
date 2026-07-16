"""
cmltrack.weave — LLM application tracing (prototype, Weave-style).

Records nested spans (chains, LLM calls, tools) with token counts, cost, and
latency into a trace tree that the UI renders. Zero dependencies.

    import cmltrack.weave as weave
    tr = weave.trace("chat", project="assistant")
    with tr.span("agent", "chain") as root:
        with tr.span("retrieve", "tool", root) as s:
            s.set_io(inputs={"q": q}, outputs={"docs": docs})
        with tr.span("llm", "llm", root) as s:
            s.set_llm("claude-opus-4-8", tokens_in=512, tokens_out=128)
    tr.finish()
"""
import json
import os
import time
import itertools

_c = itertools.count()

# illustrative $/1k tokens (input, output)
PRICING = {
    "claude-opus-4-8": (0.015, 0.075),
    "claude-sonnet-4-6": (0.003, 0.015),
    "gpt-4o": (0.005, 0.015),
    "default": (0.001, 0.002),
}


class Span:
    def __init__(self, trace, name, kind, parent):
        self.trace = trace
        self.id = f"s{next(_c)}"
        self.name = name
        self.kind = kind          # chain | llm | tool | retriever
        self.parent = parent
        self.start = time.time()
        self.end = None
        self.inputs = self.outputs = self.model = None
        self.tokens_in = self.tokens_out = 0
        self.cost = 0.0

    def set_io(self, inputs=None, outputs=None):
        if inputs is not None:
            self.inputs = inputs
        if outputs is not None:
            self.outputs = outputs
        return self

    def set_llm(self, model, tokens_in, tokens_out):
        self.model = model
        self.tokens_in, self.tokens_out = tokens_in, tokens_out
        pin, pout = PRICING.get(model, PRICING["default"])
        self.cost = tokens_in / 1000 * pin + tokens_out / 1000 * pout
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.end = time.time()
        self.trace._emit(self)


class Trace:
    def __init__(self, root, name, project):
        self.id = f"{int(time.time())}-{os.getpid()}-{next(_c)}"
        self.name, self.project = name, project
        self.spans = []
        self.dir = os.path.join(root, "traces")
        os.makedirs(self.dir, exist_ok=True)

    def span(self, name, kind="chain", parent=None):
        return Span(self, name, kind, parent.id if isinstance(parent, Span) else parent)

    def _emit(self, s):
        self.spans.append({
            "id": s.id, "parent": s.parent, "name": s.name, "kind": s.kind,
            "inputs": s.inputs, "outputs": s.outputs, "model": s.model,
            "tokens_in": s.tokens_in, "tokens_out": s.tokens_out,
            "cost": round(s.cost, 6), "latency_ms": round((s.end - s.start) * 1000, 1),
            "start": s.start,
        })

    def finish(self):
        json.dump({
            "id": self.id, "name": self.name, "project": self.project, "spans": self.spans,
            "total_cost": round(sum(s["cost"] for s in self.spans), 6),
            "tokens_in": sum(s["tokens_in"] for s in self.spans),
            "tokens_out": sum(s["tokens_out"] for s in self.spans),
            "latency_ms": round(sum(s["latency_ms"] for s in self.spans if s["parent"] is None), 1),
            "created": time.time(),
        }, open(os.path.join(self.dir, self.id + ".json"), "w"), indent=2)


def trace(name, project="llm-app", root=None):
    root = root or os.environ.get("CML_EXP_DIR", ".cml/experiments")
    return Trace(root, name, project)
