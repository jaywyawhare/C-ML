"""Newly exposed Tensor methods (permute-free movement/linalg ops) must match
NumPy. These widen the Python surface toward torch-parity; each is checked
against its NumPy reference so the binding can't silently diverge."""
import numpy as np
import pytest

import cml


@pytest.fixture(scope="module", autouse=True)
def _init():
    cml.init()


def _t(a):
    return cml.Tensor(np.asarray(a, dtype=np.float32))


def test_tril_triu():
    A = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    a = _t(A)
    for k in (-1, 0, 2):
        assert np.allclose(a.tril(k).numpy(), np.tril(A, k)), f"tril k={k}"
        assert np.allclose(a.triu(k).numpy(), np.triu(A, k)), f"triu k={k}"


def test_trace():
    A = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    assert np.allclose(_t(A).trace().numpy(), np.trace(A))


def test_diag_extract_and_build():
    A = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    assert np.allclose(_t(A).diag().numpy().ravel(), np.diag(A))
    assert np.allclose(_t(A).diag(1).numpy().ravel(), np.diag(A, 1))
    v = np.array([5, 6, 7], dtype=np.float32)
    assert np.allclose(_t(v).diag().numpy(), np.diag(v))  # 1-D -> diagonal matrix


def test_repeat_and_tile():
    v = np.array([1, 2, 3], dtype=np.float32)
    assert np.allclose(_t(v).repeat(3).numpy(), np.tile(v, 3))
    assert np.allclose(_t(v).tile(2).numpy(), np.tile(v, 2))
    A = np.arange(4, dtype=np.float32).reshape(2, 2)
    assert np.allclose(_t(A).repeat(2, 1).numpy(), np.tile(A, (2, 1)))


def test_gather_along_dim0():
    A = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    idx = [2, 0, 1]
    got = _t(A).gather(0, _t(idx)).numpy()
    assert np.allclose(got, A[idx])


def test_permute():
    A = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a = _t(A)
    for axes in ((2, 0, 1), (0, 2, 1), (2, 1, 0)):
        assert np.allclose(a.permute(*axes).numpy(), np.transpose(A, axes)), f"permute {axes}"


def test_expand():
    B = np.arange(3, dtype=np.float32).reshape(1, 3)
    assert np.allclose(_t(B).expand(4, 3).numpy(), np.broadcast_to(B, (4, 3)))
    C = np.arange(2, dtype=np.float32).reshape(2, 1)
    assert np.allclose(_t(C).expand(2, 5).numpy(), np.broadcast_to(C, (2, 5)))


def test_log_softmax():
    A = np.random.rand(3, 5).astype(np.float32)
    a = _t(A)
    ref = A - np.log(np.sum(np.exp(A), axis=1, keepdims=True))
    assert np.allclose(a.log_softmax(1).numpy(), ref, atol=1e-5)


def test_norm_full_and_dim():
    A = np.random.rand(4, 6).astype(np.float32)
    a = _t(A)
    assert np.allclose(a.norm().numpy().ravel()[0], np.linalg.norm(A), atol=1e-4)
    assert np.allclose(a.norm(2, 1).numpy().ravel(), np.linalg.norm(A, axis=1), atol=1e-4)


def test_outer():
    v = np.array([1, 2, 3], dtype=np.float32)
    w = np.array([4, 5], dtype=np.float32)
    assert np.allclose(_t(v).outer(_t(w)).numpy(), np.outer(v, w))


def test_hyperbolic_and_trunc():
    import torch
    A = (np.random.rand(3, 4).astype(np.float32) * 2 - 1)
    Ap = np.random.rand(3, 4).astype(np.float32) + 1  # >1 for acosh
    a, ap = _t(A), _t(Ap)
    assert np.allclose(a.sinh().numpy(), np.sinh(A), atol=1e-4)
    assert np.allclose(a.cosh().numpy(), np.cosh(A), atol=1e-4)
    assert np.allclose(a.asinh().numpy(), np.arcsinh(A), atol=1e-4)
    assert np.allclose(ap.acosh().numpy(), np.arccosh(Ap), atol=1e-4)
    assert np.allclose(a.atanh().numpy(), np.arctanh(A), atol=1e-4)
    assert np.allclose(a.trunc().numpy(), np.trunc(A), atol=1e-4)
    assert np.allclose(a.erfc().numpy(), torch.special.erfc(torch.from_numpy(A)).numpy(), atol=1e-4)


def test_predicates():
    B = np.array([[1.0, np.nan, np.inf], [-np.inf, 2.0, 0.0]], dtype=np.float32)
    b = _t(B)
    assert np.allclose(b.isnan().numpy(), np.isnan(B))
    assert np.allclose(b.isinf().numpy(), np.isinf(B))
    assert np.allclose(b.isfinite().numpy(), np.isfinite(B))
    C = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]], dtype=np.float32)
    assert np.allclose(_t(C).logical_not().numpy(), np.logical_not(C))


def test_activations_vs_torch():
    import torch
    import torch.nn.functional as F
    A = np.random.rand(3, 4).astype(np.float32) * 2 - 1
    a, T = _t(A), torch.from_numpy(A)
    assert np.allclose(a.gelu().numpy(), F.gelu(T, approximate="tanh").numpy(), atol=1e-4)
    assert np.allclose(a.quick_gelu().numpy(), (T * torch.sigmoid(1.702 * T)).numpy(), atol=1e-4)
    assert np.allclose(a.relu6().numpy(), F.relu6(T).numpy(), atol=1e-4)
    assert np.allclose(a.hard_sigmoid().numpy(), F.hardsigmoid(T).numpy(), atol=1e-4)
    assert np.allclose(a.hard_tanh().numpy(), F.hardtanh(T).numpy(), atol=1e-4)
    assert np.allclose(a.softplus().numpy(), F.softplus(T).numpy(), atol=1e-4)
    assert np.allclose(a.softsign().numpy(), F.softsign(T).numpy(), atol=1e-4)
    assert np.allclose(a.logsigmoid().numpy(), F.logsigmoid(T).numpy(), atol=1e-4)
    assert np.allclose(a.celu(1.0).numpy(), F.celu(T, 1.0).numpy(), atol=1e-4)


def test_elementwise_min_max_and_masked_fill():
    X = np.random.rand(3, 4).astype(np.float32)
    Y = np.random.rand(3, 4).astype(np.float32)
    x, y = _t(X), _t(Y)
    assert np.allclose(x.minimum(y).numpy(), np.minimum(X, Y))
    assert np.allclose(x.maximum(y).numpy(), np.maximum(X, Y))
    M = (X > 0.5)
    assert np.allclose(x.masked_fill(_t(M.astype(np.float32)), -1.0).numpy(), np.where(M, -1.0, X))


def test_repeat_interleave_diagonal_cummaxmin():
    V = np.array([1, 2, 3], dtype=np.float32)
    assert np.allclose(_t(V).repeat_interleave(2, 0).numpy(), np.repeat(V, 2))
    D = np.arange(12, dtype=np.float32).reshape(3, 4)
    d = _t(D)
    assert np.allclose(d.diagonal(0, 0, 1).numpy(), np.diagonal(D, 0, 0, 1))
    assert np.allclose(d.diagonal(1, 0, 1).numpy(), np.diagonal(D, 1, 0, 1))
    W = np.array([[1., 3., 2., 5., 4.]], dtype=np.float32)
    w = _t(W)
    assert np.allclose(w.cummax(1).numpy(), np.maximum.accumulate(W, axis=1))
    assert np.allclose(w.cummin(1).numpy(), np.minimum.accumulate(W, axis=1))
