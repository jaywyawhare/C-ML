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
