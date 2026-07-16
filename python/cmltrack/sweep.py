"""
cmltrack.sweep — Bayesian hyperparameter optimization (prototype, stdlib-only).

A kernel-regression surrogate + UCB acquisition (a lightweight GP stand-in that
needs no numpy/scipy). Proposes configs that balance exploiting good regions and
exploring unseen ones — a real strategy beyond grid/random.

    from cmltrack.sweep import bayes_search
    space = {"lr": (1e-4, 1e-1, "log"), "dropout": (0.0, 0.5), "opt": ["adam", "sgd"]}
    best = bayes_search(space, objective, n_trials=20)
"""
import math
import random


def _dims(space):
    d = []
    for k, dom in space.items():
        if isinstance(dom, (tuple, list)) and len(dom) >= 2 and isinstance(dom[0], (int, float)):
            log = len(dom) > 2 and dom[2] == "log"
            d.append((k, "num", dom[0], dom[1], log))
        else:
            d.append((k, "cat", list(dom), None, None))
    return d


def _vec(cfg, dims):
    v = []
    for (k, kind, a, b, log) in dims:
        if kind == "num":
            x = cfg[k]
            if log:
                v.append((math.log10(x) - math.log10(a)) / (math.log10(b) - math.log10(a) or 1))
            else:
                v.append((x - a) / (b - a) if b > a else 0.0)
        else:
            idx = a.index(cfg[k]) if cfg[k] in a else 0
            v.append(idx / max(1, len(a) - 1))
    return v


def _sample(space, rng):
    c = {}
    for k, dom in space.items():
        if isinstance(dom, (tuple, list)) and len(dom) >= 2 and isinstance(dom[0], (int, float)):
            lo, hi = dom[0], dom[1]
            if len(dom) > 2 and dom[2] == "log":
                c[k] = 10 ** rng.uniform(math.log10(lo), math.log10(hi))
            else:
                c[k] = rng.uniform(lo, hi)
        else:
            c[k] = rng.choice(list(dom))
    return c


def bayes_search(space, objective, n_trials, n_init=4, kappa=1.6, length=0.18,
                 candidates=250, seed=0, verbose=True):
    """Maximize `objective(trial_index, config) -> score`. Returns (best_score, best_config)."""
    rng = random.Random(seed)
    dims = _dims(space)
    obs = []  # (vector, score)
    best = (-1e18, None)

    def d2(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def acquisition(cfg):
        x = _vec(cfg, dims)
        ws = [math.exp(-d2(x, ox) / length) for ox, _ in obs]
        sw = sum(ws)
        if sw < 1e-9:
            return 1e18  # totally unexplored -> explore
        mean = sum(w * s for w, (_, s) in zip(ws, obs)) / sw
        uncertainty = 1.0 / (1.0 + sw)  # far from observations -> high
        return mean + kappa * uncertainty

    for i in range(n_trials):
        if i < n_init or not obs:
            cfg = _sample(space, rng)
            how = "init"
        else:
            cands = [_sample(space, rng) for _ in range(candidates)]
            cfg = max(cands, key=acquisition)
            how = "bayes"
        score = objective(i, cfg)
        obs.append((_vec(cfg, dims), score))
        if score > best[0]:
            best = (score, cfg)
        if verbose:
            print(f"  [{i:2d}/{n_trials}] {how:5s} score={score:.4f} best={best[0]:.4f} cfg={ {k: (round(v,4) if isinstance(v,float) else v) for k,v in cfg.items()} }")
    return best
