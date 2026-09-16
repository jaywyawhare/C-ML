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
