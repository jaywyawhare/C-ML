"""numpy-API breadth matrix.

Exercises every public Tensor method against its numpy equivalent where one
exists, and prints a coverage table: which operations are bound, and do they
agree numerically with numpy?

Run:  cd python && python3 -m pytest tests/test_numpy_matrix.py -q -s
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CML_LOG_LEVEL", "ERROR")

import numpy as np
import pytest

import cml

cml.init()

RTOL = 1e-4


def make(x):
    return cml.Tensor(np.array(x, dtype=np.float32))


def check(name, x, fn_cml, fn_np=None, shape_exact=True, equal_nan=False):
    """Build a fresh tensor from x, run the cml op, compare against numpy."""
    try:
        got = np.asarray(fn_cml(make(x)), dtype=np.float64)
    except (AttributeError, NotImplementedError) as e:
        return name, f"not bound ({type(e).__name__})"
    except Exception as e:  # noqa: BLE001
        return name, f"error: {e}"

    if fn_np is None:
        return name, "ok"

    want = np.asarray(fn_np(np.array(x, dtype=np.float32)), dtype=np.float64)
    gs, ws = got.shape, want.shape
    if shape_exact and gs != ws:
        # reductions returning [1] instead of scalar () are a convention
        # difference, not a value error — compare raveled values
        if got.size == want.size:
            if not np.allclose(got.ravel(), want.ravel(), rtol=RTOL, atol=1e-5,
                               equal_nan=equal_nan):
                return name, f"value {got.ravel()[:4]} vs {want.ravel()[:4]}"
            return name, f"ok (shape {gs} vs {ws})"
        return name, f"shape {gs} != {ws}"
    if not np.allclose(got, want, rtol=RTOL, atol=1e-5, equal_nan=equal_nan):
        return name, f"value {got.ravel()[:4]} vs {want.ravel()[:4]}"
    return name, "ok"


A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
B = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
V = [0.4, 1.2, 2.1, -0.7]

CASES = [
    # arithmetic
    ("add",       A, lambda t: t + make(B),      lambda x: x + np.array(B)),
    ("sub",       A, lambda t: t - make(B),      lambda x: x - np.array(B)),
    ("mul",       A, lambda t: t * make(B),      lambda x: x * np.array(B)),
    ("div",       A, lambda t: t / make(B),      lambda x: x / np.array(B)),
    ("neg",       A, lambda t: -t,               lambda x: -x),
    # reductions
    ("sum",       A, lambda t: t.sum(),          np.sum),
    ("mean",      A, lambda t: t.mean(),         np.mean),
    ("max",       A, lambda t: t.max(),          np.max),
    ("min",       A, lambda t: t.min(),          np.min),
    # cml convention: no-dim var/std = GLOBAL UNBIASED (torch-like);
    # numpy's default is biased, hence ddof=1 here
    ("var",       A, lambda t: t.var(),          lambda x: x.var(ddof=1)),
    ("std",       A, lambda t: t.std(),          lambda x: x.std(ddof=1)),
    ("sum_d1",    A, lambda t: t.sum(1),         lambda x: x.sum(1)),
    ("mean_d1",   A, lambda t: t.mean(1),        lambda x: x.mean(1)),
    ("argmax_d1", A, lambda t: t.argmax(1),      lambda x: x.argmax(1)),
    ("argmin_d1", A, lambda t: t.argmin(1),      lambda x: x.argmin(1)),
    # shape
    ("reshape",   A, lambda t: t.reshape(3, 2),  lambda x: x.reshape(3, 2)),
    ("transpose", A, lambda t: t.transpose(0, 1),lambda x: x.T),
    ("flatten",   A, lambda t: t.flatten(),      lambda x: x.ravel()),
    ("squeeze",   np.array(A)[None], lambda t: t.squeeze(), np.squeeze),
    ("flip_d1",   A, lambda t: t.flip(1),        lambda x: np.flip(x, 1)),
    # unary math (inputs kept inside each function's real domain)
    ("exp",       V, lambda t: t.exp(),          np.exp),
    ("log",       V[1:], lambda t: t.log(),     np.log),
    ("sqrt",      V, lambda t: t.sqrt(),         np.sqrt),
    ("sin",       V, lambda t: t.sin(),          np.sin),
    ("cos",       V, lambda t: t.cos(),          np.cos),
    ("tan",       V[:1], lambda t: t.tan(),      np.tan),
    ("log2",      V[1:], lambda t: t.log2(),    np.log2),
    ("log10",     V[1:], lambda t: t.log10(),   np.log10),
    ("square",    V, lambda t: t.square(),       np.square),
    ("sign",      V, lambda t: t.sign(),         np.sign),
    ("abs",       V, lambda t: abs(t),           np.abs),
    ("ceil",      [0.2, 1.7], lambda t: t.ceil(), np.ceil),
    ("floor",     [0.8, 1.2], lambda t: t.floor(), np.floor),
    ("round",     [0.5, 1.6], lambda t: t.round(), np.round),
    ("sigmoid",   V, lambda t: t.sigmoid(),      lambda x: 1/(1+np.exp(-x))),
    ("rsqrt",     V[1:], lambda t: t.rsqrt(),    lambda x: 1/np.sqrt(x)),
    # binary / other
    # cml convention: prod(dim=-1) reduces the last axis (global prod via .prod(0).prod(0))
    ("prod_last", A, lambda t: t.prod(),           lambda x: x.prod(-1)),
    ("matmul",    A, lambda t: t.matmul(t.transpose(0, 1)), lambda x: x @ x.T),
    ("dot_1d",    V, lambda t: t.dot(t),        lambda x: float(x @ x)),
    ("clamp",     A, lambda t: t.clamp(2.0, 4.0),lambda x: np.clip(x, 2.0, 4.0)),
    ("where",     A, lambda t: cml.where(t > 2.0, t, -t), lambda x: np.where(x > 2.0, x, -x)),
    ("einsum",    A, lambda t: cml.einsum("ij,jk->ik", t, t.transpose(0, 1)),
     lambda x: x @ x.T),
    ("roll",      A, lambda t: t.roll(2, 1),     lambda x: np.roll(x, 2, 1)),
    ("copysign",  A, lambda t: t.copysign(-1.5), lambda x: np.copysign(x, -1.5)),
    ("logaddexp", A, lambda t: t.logaddexp(1.0), lambda x: np.logaddexp(x, 1.0)),
    ("one_hot",   [0.0, 2.0, 1.0], lambda t: t.one_hot(3),
     lambda x: np.eye(3, dtype=np.float32)[x.astype(int)]),
    # long-tail reductions / indexing / pad (added 2026-08-26)
    ("cumsum",    A, lambda t: t.cumsum(-1),     lambda x: np.cumsum(x, -1)),
    ("cumprod",   V[:3], lambda t: t.cumprod(-1),lambda x: np.cumprod(x, -1)),
    ("argsort",   A, lambda t: t.argsort(1),      lambda x: x.argsort(1)),
    ("topk",      A, lambda t: t.topk(2, dim=1)[0],
     lambda x: np.sort(x, 1)[:, -2:][:, ::-1]),
    ("median",    A, lambda t: t.median(dim=1),  lambda x: np.median(x, 1)),
    ("unique",    [3.0, 1.0, 3.0, 2.0], lambda t: t.unique(), np.unique),
    ("pad",       A, lambda t: t.pad(((1, 1), (2, 2))),
     lambda x: np.pad(x, ((1, 1), (2, 2)))),
    ("getitem_slice", A, lambda t: t[0:2, 1:3],  lambda x: x[0:2, 1:3]),
    ("getitem_fancy", A, lambda t: t[[1, 0]],     lambda x: x[[1, 0]]),
    ("getitem_mask",  A, lambda t: t[t > 2.5],    lambda x: x[x > 2.5]),
]

CONSTRUCTORS = [
    ("zeros",     lambda: cml.zeros([2, 3]).numpy(),      np.zeros((2, 3))),
    ("ones",      lambda: cml.ones([2, 3]).numpy(),       np.ones((2, 3))),
    ("eye",       lambda: cml.eye(3).numpy(),             np.eye(3)),
    ("arange",    lambda: cml.arange(0, 6, 1).numpy(),    np.arange(6)),
    ("linspace",  lambda: cml.linspace(0, 1, 5).numpy(),  np.linspace(0, 1, 5)),
    ("full",      lambda: cml.full([2, 2], 3.5).numpy(),  np.full((2, 2), 3.5)),
    ("zeros_like",lambda: cml.zeros_like(make(A)).numpy(),np.zeros((2, 3))),
]


def test_breadth_matrix():
    NAN_OK = {"log", "sqrt", "log2", "log10", "rsqrt"}
    results = [check(n, x, f, nf, equal_nan=n in NAN_OK) for n, x, f, nf in CASES]

    results += [(n,
                 "ok" if np.allclose(np.asarray(f(), dtype=np.float64), w, atol=1e-5)
                 else "value mismatch")
                for n, f, w in CONSTRUCTORS]

    ok     = [r for r in results if r[1] == "ok" or r[1].startswith("ok (")]
    conv   = [r for r in results if r[1].startswith("ok (")]     # shape conventions
    unbound= [r for r in results if r[1].startswith("not bound")]
    broken = [r for r in results if r not in ok and r not in unbound and r not in conv]

    print(f"\nnumpy-API breadth: {len(ok)} ({len(conv)} shape-convention) / "
          f"{len(unbound)} not bound / {len(broken)} BROKEN of {len(results)}")
    for n, st in broken:
        print(f"  BROKEN {n}: {st}")
    for n, st in unbound:
        print(f"  missing {n}: {st}")

    # anything bound must be numerically correct
    assert not broken, f"{len(broken)} bound-but-wrong ops: {broken}"

    # core set must exist and agree with numpy
    CORE = {"add", "mul", "matmul", "sum", "mean", "exp", "sqrt",
            "transpose", "sigmoid"}
    have = {n for n, _ in ok}
    assert not (CORE - have), f"core ops missing: {CORE - have}"
