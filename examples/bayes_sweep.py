#!/usr/bin/env python3
"""
bayes_sweep.py — Bayesian hyperparameter optimization via the cmltrack SDK.

Each trial is a real tracked run (scalars, weight histograms via .watch(),
system/GPU telemetry, a PR curve). The optimizer proposes configs with a
kernel-surrogate + UCB acquisition, converging on the good region faster than
random. Demonstrates the Python SDK end to end.

    python3 examples/bayes_sweep.py
    python3 viz/serve.py   ->  http://localhost:6969
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import cmltrack as ct
from cmltrack.sweep import bayes_search


def quality(cfg):
    """Simulated 'true' ceiling accuracy — peaked at lr~0.01, dropout~0.2, hidden~48, adam."""
    dl = abs(math.log10(cfg["lr"]) - math.log10(0.01)) / 2
    dd = abs(cfg["dropout"] - 0.2) / 0.5
    dh = abs(cfg["hidden"] - 48) / 48
    opt_penalty = 0.0 if cfg["opt"] == "adam" else 0.06
    return max(0.40, 0.99 - 0.35 * (0.6 * dl + 0.2 * dd + 0.2 * dh) - opt_penalty)


def objective(i, cfg):
    run = ct.init(project="bo-search", name=f"bo-{i}", config=cfg, sweep="bo-sweep-1", tags=["bayes"])
    rng = random.Random(1000 + i)
    ceiling, best = quality(cfg), 0.0
    W = [rng.gauss(0, 0.5) for _ in range(128)]
    for e in range(1, 41):
        acc = max(0.0, min(1.0, ceiling * (1 - math.exp(-0.09 * e)) + 0.01 * (rng.random() - 0.5)))
        best = max(best, acc)
        run.log({"val/accuracy": acc, "train/loss": (1 - acc) * 0.9 + 0.02}, step=e)
        W = [w - cfg["lr"] * rng.gauss(0, 1) * (1 - acc) for w in W]
        if e % 10 == 0:
            run.watch({"weights/layer0": W}, step=e)
            run.log_system(e)
    xs = [k / 10 for k in range(11)]
    run.log_curve("pr_curve", xs, [ceiling * (1 - 0.5 * x) for x in xs], "recall", "precision")
    run.summary["best_val_accuracy"] = best
    run.config["best_val_accuracy"] = round(best, 4)
    if ceiling < 0.6:
        run.alert("warn", "Sampled a low-quality hyperparameter region")
    run.finish()
    return best


if __name__ == "__main__":
    space = {"lr": (1e-4, 1e-1, "log"), "dropout": (0.0, 0.5), "hidden": (16, 64), "opt": ["adam", "sgd"]}
    print("Bayesian sweep (kernel surrogate + UCB acquisition):")
    best_score, best_cfg = bayes_search(space, objective, n_trials=16, n_init=4, seed=7)
    pretty = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best_cfg.items()}
    print(f"\nBest: score={best_score:.4f}  cfg={pretty}")
    print("View: python3 viz/serve.py  ->  http://localhost:6969")
