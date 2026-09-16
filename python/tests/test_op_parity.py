"""Op-level forward + backward parity against PyTorch.

For each op we run the same computation in C-ML and torch on identical inputs,
then compare the forward output and the input gradient (via a scalar sum().
backward()). This is the credibility keystone: every op that claims to be
differentiable must match torch to tight tolerance on both passes.

Run:  cd python && PYTHONPATH=. python -m pytest tests/test_op_parity.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CML_LOG_LEVEL", "ERROR")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import cml

cml.init()

TOL = 2e-4


def _check(name, x_np, cml_fn, torch_fn):
    """Compare forward + input-grad of a single-input op."""
    xc = cml.Tensor(x_np.copy())
    xc.requires_grad_(True)
    yc = cml_fn(xc)
    fwd_c = np.asarray(yc.numpy())
    yc.sum().backward()
    gc = np.asarray(xc.grad.numpy())

    xt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    yt = torch_fn(xt)
    fwd_t = yt.detach().numpy()
    yt.sum().backward()
    gt = xt.grad.detach().numpy()

    assert np.allclose(fwd_c, fwd_t, atol=TOL, rtol=TOL), \
        f"{name}: forward mismatch (max {np.abs(fwd_c - fwd_t).max():.3e})"
    assert np.allclose(gc, gt, atol=TOL, rtol=TOL), \
        f"{name}: grad mismatch (max {np.abs(gc - gt).max():.3e})"


rs = np.random.RandomState(7)
X = rs.randn(4, 5).astype(np.float32)
XPOS = np.abs(X) + 0.5  # strictly positive, for log/sqrt


@pytest.mark.parametrize("name,cf,tf,inp", [
    ("relu",    lambda t: t.relu(),    torch.relu,               "X"),
    ("sigmoid", lambda t: t.sigmoid(), torch.sigmoid,            "X"),
    ("tanh",    lambda t: t.tanh(),    torch.tanh,               "X"),
    ("exp",     lambda t: t.exp(),     torch.exp,                "X"),
    ("neg",     lambda t: -t,          lambda t: -t,             "X"),
    ("square",  lambda t: t.square(),  lambda t: t * t,          "X"),
    ("log",     lambda t: t.log(),     torch.log,                "XPOS"),
    ("sqrt",    lambda t: t.sqrt(),    torch.sqrt,               "XPOS"),
])
def test_unary_parity(name, cf, tf, inp):
    _check(name, XPOS if inp == "XPOS" else X, cf, tf)


def test_binary_parity():
    a = rs.randn(4, 5).astype(np.float32)
    b = rs.randn(4, 5).astype(np.float32)
    for name, cf, tf in [
        ("add", lambda x, y: x + y, lambda x, y: x + y),
        ("sub", lambda x, y: x - y, lambda x, y: x - y),
        ("mul", lambda x, y: x * y, lambda x, y: x * y),
    ]:
        ac = cml.Tensor(a.copy()); ac.requires_grad_(True)
        bc = cml.Tensor(b.copy()); bc.requires_grad_(True)
        oc = cf(ac, bc); oc.sum().backward()
        at = torch.tensor(a, requires_grad=True)
        bt = torch.tensor(b, requires_grad=True)
        ot = tf(at, bt); ot.sum().backward()
        assert np.allclose(np.asarray(oc.numpy()), ot.detach().numpy(), atol=TOL), name
        assert np.allclose(np.asarray(ac.grad.numpy()), at.grad.numpy(), atol=TOL), f"{name} da"
        assert np.allclose(np.asarray(bc.grad.numpy()), bt.grad.numpy(), atol=TOL), f"{name} db"


def test_matmul_parity():
    a = rs.randn(4, 6).astype(np.float32)
    b = rs.randn(6, 3).astype(np.float32)
    ac = cml.Tensor(a.copy()); ac.requires_grad_(True)
    bc = cml.Tensor(b.copy()); bc.requires_grad_(True)
    oc = ac.matmul(bc); oc.sum().backward()
    at = torch.tensor(a, requires_grad=True)
    bt = torch.tensor(b, requires_grad=True)
    ot = at @ bt; ot.sum().backward()
    assert np.allclose(np.asarray(oc.numpy()), ot.detach().numpy(), atol=TOL)
    assert np.allclose(np.asarray(ac.grad.numpy()), at.grad.numpy(), atol=TOL), "matmul da"
    assert np.allclose(np.asarray(bc.grad.numpy()), bt.grad.numpy(), atol=TOL), "matmul db"


def test_reduction_parity():
    for name, cf, tf in [
        ("sum",  lambda t: t.sum(),  lambda t: t.sum()),
        ("mean", lambda t: t.mean(), lambda t: t.mean()),
    ]:
        _check(name, X, cf, tf)


def test_shape_parity():
    # reshape (the op that severed the graph before the fix)
    _check("reshape", X, lambda t: t.reshape(2, 10), lambda t: t.reshape(2, 10))
    # transpose
    _check("transpose", X, lambda t: t.transpose(0, 1), lambda t: t.transpose(0, 1))


def test_softmax_parity():
    _check("softmax", X,
           lambda t: t.softmax(1),
           lambda t: torch.softmax(t, dim=1))


# ── broadened coverage ────────────────────────────────────────────────────
@pytest.mark.parametrize("name,cf,tf,inp", [
    ("reciprocal", lambda t: t.reciprocal(), torch.reciprocal, "XPOS"),
    ("rsqrt",      lambda t: t.rsqrt(),      torch.rsqrt,      "XPOS"),
    ("sin",        lambda t: t.sin(),        torch.sin,        "X"),
    ("cos",        lambda t: t.cos(),        torch.cos,        "X"),
    ("abs",        lambda t: abs(t),         torch.abs,        "X"),
])
def test_more_unary_parity(name, cf, tf, inp):
    _check(name, XPOS if inp == "XPOS" else X, cf, tf)


def test_clamp_parity():
    _check("clamp", X, lambda t: t.clamp(-0.5, 0.5),
           lambda t: t.clamp(-0.5, 0.5))


def test_pow_parity():
    _check("pow3", XPOS, lambda t: t.pow(3.0), lambda t: t.pow(3.0))


def test_axis_reduction_parity():
    X3 = rs.randn(2, 3, 4).astype(np.float32)
    _check("sum_axis1",  X3, lambda t: t.sum(dim=1),  lambda t: t.sum(dim=1))
    _check("mean_axis2", X3, lambda t: t.mean(dim=2), lambda t: t.mean(dim=2))


def test_max_min_reduction_parity():
    _check("max_axis1", X, lambda t: t.max(dim=1), lambda t: t.max(dim=1).values)
    _check("min_axis1", X, lambda t: t.min(dim=1), lambda t: t.min(dim=1).values)


def test_flatten_parity():
    X3 = rs.randn(2, 3, 4).astype(np.float32)
    _check("flatten", X3, lambda t: t.flatten(1), lambda t: torch.flatten(t, 1))


def test_batched_matmul_parity():
    a = rs.randn(3, 4, 6).astype(np.float32)
    b = rs.randn(3, 6, 5).astype(np.float32)
    ac = cml.Tensor(a.copy()); ac.requires_grad_(True)
    bc = cml.Tensor(b.copy()); bc.requires_grad_(True)
    oc = ac.matmul(bc); oc.sum().backward()
    at = torch.tensor(a, requires_grad=True)
    bt = torch.tensor(b, requires_grad=True)
    ot = at @ bt; ot.sum().backward()
    assert np.allclose(np.asarray(oc.numpy()).reshape(3, 4, 5), ot.detach().numpy(), atol=TOL), "bmm fwd"
    assert np.allclose(np.asarray(ac.grad.numpy()).reshape(a.shape), at.grad.numpy(), atol=TOL), "bmm da"
    assert np.allclose(np.asarray(bc.grad.numpy()).reshape(b.shape), bt.grad.numpy(), atol=TOL), "bmm db"


def test_cat_stack_parity():
    a = rs.randn(2, 3).astype(np.float32)
    b = rs.randn(2, 3).astype(np.float32)
    # cat
    ac = cml.Tensor(a.copy()); ac.requires_grad_(True)
    bc = cml.Tensor(b.copy()); bc.requires_grad_(True)
    oc = cml.Tensor.cat([ac, bc], 0); oc.sum().backward()
    at = torch.tensor(a, requires_grad=True); bt = torch.tensor(b, requires_grad=True)
    ot = torch.cat([at, bt], 0); ot.sum().backward()
    assert np.allclose(np.asarray(oc.numpy()).reshape(4, 3), ot.detach().numpy(), atol=TOL), "cat fwd"
    assert np.allclose(np.asarray(ac.grad.numpy()), at.grad.numpy(), atol=TOL), "cat grad"


@pytest.mark.parametrize("name,cf,tf", [
    ("floor", lambda t: t.floor(), torch.floor),
    ("ceil",  lambda t: t.ceil(),  torch.ceil),
    ("round", lambda t: t.round(), torch.round),
    ("sign",  lambda t: t.sign(),  torch.sign),
])
def test_nondiff_forward_parity(name, cf, tf):
    """Non-differentiable ops: forward value must match (grad is zero/undefined)."""
    xc = cml.Tensor(X.copy())
    fwd_c = np.asarray(cf(xc).numpy())
    fwd_t = tf(torch.tensor(X)).numpy()
    assert np.allclose(fwd_c, fwd_t, atol=TOL), \
        f"{name}: forward mismatch (max {np.abs(fwd_c - fwd_t).max():.3e})"


# ── ops that exercise distinct backward-pass paths ─────────────────────────
def test_cumsum_parity():
    X3 = rs.randn(3, 5).astype(np.float32)
    _check("cumsum", X3, lambda t: t.cumsum(1),
           lambda t: torch.cumsum(t, dim=1))


def test_prod_parity():
    # bounded away from 0 so d(prod)/dx_i = prod/x_i stays well-conditioned
    P = (rs.rand(3, 4).astype(np.float32) + 0.5)
    _check("prod", P, lambda t: t.prod(dim=1),
           lambda t: torch.prod(t, dim=1))


def test_var_std_parity():
    _check("var", X, lambda t: t.var(dim=1), lambda t: t.var(dim=1, unbiased=True))
    _check("std", X, lambda t: t.std(dim=1), lambda t: t.std(dim=1, unbiased=True))


def test_where_parity():
    a = rs.randn(4, 5).astype(np.float32)
    b = rs.randn(4, 5).astype(np.float32)
    cond = (rs.rand(4, 5) > 0.5).astype(np.float32)
    ac = cml.Tensor(a.copy()); ac.requires_grad_(True)
    bc = cml.Tensor(b.copy()); bc.requires_grad_(True)
    oc = ac.where(cml.Tensor(cond.copy()), bc); oc.sum().backward()
    at = torch.tensor(a, requires_grad=True); bt = torch.tensor(b, requires_grad=True)
    ot = torch.where(torch.tensor(cond) > 0, at, bt); ot.sum().backward()
    assert np.allclose(np.asarray(oc.numpy()), ot.detach().numpy(), atol=TOL), "where fwd"
    assert np.allclose(np.asarray(ac.grad.numpy()), at.grad.numpy(), atol=TOL), "where da"
    assert np.allclose(np.asarray(bc.grad.numpy()), bt.grad.numpy(), atol=TOL), "where db"
