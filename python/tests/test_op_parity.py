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
import torch.nn.functional as F

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


def _check_bin(name, a_np, b_np, cml_fn, torch_fn):
    """Compare forward + both input grads of a two-input op (broadcasting aware)."""
    ac = cml.Tensor(a_np.copy()); ac.requires_grad_(True)
    bc = cml.Tensor(b_np.copy()); bc.requires_grad_(True)
    oc = cml_fn(ac, bc)
    fwd_c = np.asarray(oc.numpy())
    oc.sum().backward()
    ga_c = np.asarray(ac.grad.numpy())
    gb_c = np.asarray(bc.grad.numpy())

    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    ot = torch_fn(at, bt)
    ot.sum().backward()

    assert np.allclose(fwd_c.reshape(ot.shape), ot.detach().numpy(), atol=TOL, rtol=TOL), \
        f"{name}: forward mismatch"
    assert np.allclose(ga_c.reshape(a_np.shape), at.grad.numpy(), atol=TOL, rtol=TOL), \
        f"{name}: grad-a mismatch"
    assert np.allclose(gb_c.reshape(b_np.shape), bt.grad.numpy(), atol=TOL, rtol=TOL), \
        f"{name}: grad-b mismatch"


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
        _check_bin(name, a, b, cf, tf)


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
    # flip (also severed the graph before its uop_flip fix)
    _check("flip0", X, lambda t: t.flip(0), lambda t: torch.flip(t, [0]))
    _check("flip1", X, lambda t: t.flip(1), lambda t: torch.flip(t, [1]))


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


def test_roll_parity():
    _check("roll", X, lambda t: t.roll(2, 1), lambda t: torch.roll(t, 2, 1))


def test_cumprod_parity():
    P = (rs.rand(3, 4).astype(np.float32) + 0.3)
    _check("cumprod", P, lambda t: t.cumprod(1), lambda t: torch.cumprod(t, 1))


def test_softmax_dims_parity():
    X3 = rs.randn(2, 3, 4).astype(np.float32)
    _check("softmax0", X3, lambda t: t.softmax(0), lambda t: torch.softmax(t, 0))
    _check("softmax1", X3, lambda t: t.softmax(1), lambda t: torch.softmax(t, 1))


def test_keepdim_transpose3d_parity():
    X3 = rs.randn(2, 3, 4).astype(np.float32)
    _check("sum_keepdim", X3, lambda t: t.sum(1, True), lambda t: t.sum(1, keepdim=True))
    _check("transpose3d", X3, lambda t: t.transpose(0, 2), lambda t: t.transpose(0, 2))


def test_masked_select_parity():
    a = rs.randn(4, 5).astype(np.float32)
    mask = (rs.rand(4, 5) > 0.4).astype(np.float32)
    ac = cml.Tensor(a.copy()); ac.requires_grad_(True)
    oc = ac.masked_select(cml.Tensor(mask.copy())); oc.sum().backward()
    at = torch.tensor(a, requires_grad=True)
    ot = torch.masked_select(at, torch.tensor(mask) > 0); ot.sum().backward()
    assert np.allclose(np.sort(np.asarray(oc.numpy()).ravel()),
                       np.sort(ot.detach().numpy()), atol=TOL), "masked_select fwd"
    assert np.allclose(np.asarray(ac.grad.numpy()).reshape(a.shape),
                       at.grad.numpy(), atol=TOL), "masked_select grad"


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


# ── parity at scale ────────────────────────────────────────────────────────
# Sweep the proven differentiable ops over a grid of ranks and broadcast-prone
# shapes, checking forward AND input-grad against torch for every combination.
SWEEP_SHAPES = [(1,), (7,), (3, 4), (1, 5), (5, 1),
                (2, 3, 4), (2, 1, 4), (1, 3, 1), (2, 3, 4, 5)]

_GEN_UNARY = [
    ("relu", lambda t: t.relu(), torch.relu),
    ("sigmoid", lambda t: t.sigmoid(), torch.sigmoid),
    ("tanh", lambda t: t.tanh(), torch.tanh),
    ("exp", lambda t: t.exp(), torch.exp),
    ("sin", lambda t: t.sin(), torch.sin),
    ("cos", lambda t: t.cos(), torch.cos),
    ("square", lambda t: t.square(), lambda t: t * t),
    ("sinh", lambda t: t.sinh(), torch.sinh),
    ("gelu", lambda t: t.gelu(), lambda t: F.gelu(t, approximate="tanh")),
    ("softplus", lambda t: t.softplus(), F.softplus),
    ("softsign", lambda t: t.softsign(), F.softsign),
    ("logsigmoid", lambda t: t.logsigmoid(), F.logsigmoid),
]
_POS_UNARY = [
    ("log", lambda t: t.log(), torch.log),
    ("sqrt", lambda t: t.sqrt(), torch.sqrt),
    ("rsqrt", lambda t: t.rsqrt(), torch.rsqrt),
    ("reciprocal", lambda t: t.reciprocal(), torch.reciprocal),
    ("log10", lambda t: t.log10(), torch.log10),
]


@pytest.mark.parametrize("shape", SWEEP_SHAPES)
@pytest.mark.parametrize("name,cf,tf", _GEN_UNARY)
def test_unary_shape_sweep(shape, name, cf, tf):
    _check(f"{name}{shape}", rs.randn(*shape).astype(np.float32), cf, tf)


@pytest.mark.parametrize("shape", SWEEP_SHAPES)
@pytest.mark.parametrize("name,cf,tf", _POS_UNARY)
def test_pos_unary_shape_sweep(shape, name, cf, tf):
    x = (np.abs(rs.randn(*shape)) + 0.5).astype(np.float32)
    _check(f"{name}{shape}", x, cf, tf)


# Newly bound activations: lock in their backward pass (not only forward).
@pytest.mark.parametrize("name,cf,tf,dom", [
    ("cosh", lambda t: t.cosh(), torch.cosh, "x"),
    ("asinh", lambda t: t.asinh(), torch.asinh, "x"),
    ("atanh", lambda t: t.atanh(), torch.atanh, "unit"),
    ("quick_gelu", lambda t: t.quick_gelu(), lambda t: t * torch.sigmoid(1.702 * t), "x"),
    ("relu6", lambda t: t.relu6(), F.relu6, "x"),
    ("hard_sigmoid", lambda t: t.hard_sigmoid(), F.hardsigmoid, "x"),
    ("hard_tanh", lambda t: t.hard_tanh(), F.hardtanh, "x"),
    ("celu", lambda t: t.celu(1.0), lambda t: F.celu(t, 1.0), "x"),
])
def test_new_activation_autograd(name, cf, tf, dom):
    x = np.clip(X, -0.9, 0.9) if dom == "unit" else X
    _check(name, x, cf, tf)


# Broadcasting binaries: the backward must correctly sum-reduce over the
# broadcast dimensions on each operand.
_BCAST_PAIRS = [
    ((3, 4), (4,)),
    ((3, 1), (1, 4)),
    ((2, 3, 4), (3, 4)),
    ((2, 3, 4), (1,)),
    ((2, 1, 4), (2, 3, 4)),
]


@pytest.mark.parametrize("sa,sb", _BCAST_PAIRS)
@pytest.mark.parametrize("name,cf,tf", [
    ("add", lambda x, y: x + y, lambda x, y: x + y),
    ("sub", lambda x, y: x - y, lambda x, y: x - y),
    ("mul", lambda x, y: x * y, lambda x, y: x * y),
    ("minimum", lambda x, y: x.minimum(y), torch.minimum),
    ("maximum", lambda x, y: x.maximum(y), torch.maximum),
])
def test_binary_broadcast_sweep(sa, sb, name, cf, tf):
    a = rs.randn(*sa).astype(np.float32)
    b = rs.randn(*sb).astype(np.float32)
    _check_bin(f"{name}{sa}x{sb}", a, b, cf, tf)
