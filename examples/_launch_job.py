#!/usr/bin/env python3
"""A tiny trainable job launched by the Launch agent. Also demonstrates artifact
lineage (uses a dataset artifact, produces a model artifact) and a rich media table."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import cmltrack as ct

lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
run = ct.init("launch-jobs", name=f"job-lr{lr}", config={"lr": lr}, tags=["launched"])
run.use_artifact("iris-dataset", "v1")           # lineage: input

for e in range(1, 26):
    loss = math.exp(-lr * 30 * e) + 0.02
    run.log({"train/loss": loss, "val/accuracy": 1 - loss * 0.8}, step=e)
run.summary["final_loss"] = loss

# rich media table: a couple of "predictions" with an image cell each
def swatch(r, g, b):
    return {"_image": bytes([r, g, b] * (16 * 16)), "w": 16, "h": 16}
run.log_rich_table("predictions", ["id", "pred", "conf", "thumb"], [
    {"id": 0, "pred": "setosa", "conf": 0.98, "thumb": swatch(80, 200, 120)},
    {"id": 1, "pred": "versicolor", "conf": 0.76, "thumb": swatch(200, 160, 60)},
    {"id": 2, "pred": "virginica", "conf": 0.91, "thumb": swatch(120, 120, 220)},
])

mp = f".cml/experiments/model_launch_{lr}.bin"
open(mp, "wb").write(b"\x00" * 256)
run.log_artifact("iris-model", mp, type="model", aliases="latest")   # lineage: output
run.finish()
print(f"job lr={lr} done, final_loss={loss:.4f}")
