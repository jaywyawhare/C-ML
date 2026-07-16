#!/usr/bin/env python3
"""Weave LLM-tracing demo — simulates a RAG assistant and records trace trees
(chain -> retriever + llm) with token counts, cost, and latency."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import cmltrack.weave as weave

QUESTIONS = [
    "What is C-ML?",
    "How do I train an MLP on Iris?",
    "Explain reverse-mode autograd",
    "Does the Vulkan backend support conv2d?",
]

rng = random.Random(7)
for q in QUESTIONS:
    tr = weave.trace(name=f"chat: {q[:24]}", project="cml-assistant")
    with tr.span("agent", "chain") as root:
        root.set_io(inputs={"question": q})
        with tr.span("retrieve_docs", "retriever", root) as ret:
            time.sleep(0.005)
            ret.set_io(inputs={"query": q}, outputs={"docs_found": rng.randint(2, 6)})
        with tr.span("rerank", "tool", root) as rr:
            time.sleep(0.003)
            rr.set_io(outputs={"top_k": 3})
        with tr.span("llm.generate", "llm", root) as gen:
            time.sleep(0.02 + 0.02 * rng.random())
            tin, tout = rng.randint(300, 1200), rng.randint(80, 400)
            gen.set_llm("claude-opus-4-8", tin, tout)
            gen.set_io(inputs={"prompt": f"Context + {q}"}, outputs={"answer": "…(generated)…"})
        root.set_io(outputs={"answer": "…(final answer)…"})
    tr.finish()
    print(f"traced: {q}")

print("\nView: python3 viz/serve.py  ->  http://localhost:6969  (Weave tab)")
