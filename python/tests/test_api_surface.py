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
