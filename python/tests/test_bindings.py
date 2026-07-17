"""Regression tests for the ergonomic tensor API (creation, scalar ops, autograd).

Run with:  cd python && python3 -m pytest tests/test_bindings.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CML_LOG_LEVEL", "ERROR")

import numpy as np
import pytest

import cml


def test_tensor_factory_from_nested_list():
    t = cml.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)
    np.testing.assert_allclose(t.numpy(), [[1, 2], [3, 4]])


def test_tensor_constructor_accepts_data():
    # Constructor takes data directly (not just a C pointer).
    t = cml.Tensor([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(t.numpy(), [[5, 6], [7, 8]])


def test_tensor_from_scalar_and_numpy():
    s = cml.tensor(3.0)
    assert s.shape == (1,)
    assert float(s.numpy()[0]) == 3.0
    a = cml.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    assert a.shape == (2, 3)


def test_tensor_tensor_ops():
    a = cml.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = cml.tensor([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose((a + b).numpy(), [[6, 8], [10, 12]])
    np.testing.assert_allclose((a * b).numpy(), [[5, 12], [21, 32]])
    np.testing.assert_allclose((a @ b).numpy(), [[19, 22], [43, 50]])


@pytest.mark.parametrize(
    "expr, expected",
    [
        (lambda t: t + 1.0, [[2, 3], [4, 5]]),
        (lambda t: 1.0 + t, [[2, 3], [4, 5]]),
        (lambda t: t - 2.0, [[-1, 0], [1, 2]]),
        (lambda t: 10.0 - t, [[9, 8], [7, 6]]),
        (lambda t: t * 3.0, [[3, 6], [9, 12]]),
        (lambda t: t / 2.0, [[0.5, 1.0], [1.5, 2.0]]),
        (lambda t: 12.0 / t, [[12.0, 6.0], [4.0, 3.0]]),
    ],
)
def test_scalar_broadcast_ops(expr, expected):
    t = cml.tensor([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(expr(t).numpy(), expected, rtol=1e-5)


def test_scalar_op_returns_notimplemented_for_bad_type():
    t = cml.tensor([1.0, 2.0])
    with pytest.raises(TypeError):
        _ = t + "nope"


def test_autograd_through_scalar_ops():
    # y = sum(x*2 + 1)  ->  dy/dx = 2
    x = cml.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * 2.0 + 1.0).sum()
    cml.backward(y)
    g = cml.get_grad(x)
    assert g is not None
    np.testing.assert_allclose(g.numpy(), [2.0, 2.0, 2.0])


def test_autograd_square():
    x = cml.tensor([3.0], requires_grad=True)
    y = x * x
    cml.backward(y)
    np.testing.assert_allclose(cml.get_grad(x).numpy(), [6.0])  # d(x^2)/dx = 2x


def test_numpy_roundtrip_preserves_shape_and_values():
    arr = np.random.randn(4, 5).astype(np.float32)
    t = cml.tensor(arr)
    assert t.shape == (4, 5)
    np.testing.assert_allclose(t.numpy(), arr, rtol=1e-6)


def test_nn_forward_and_optimizer_construct():
    """The nn wrappers pass concrete layer handles where the C API wants Module*;
    these used to raise a CFFI TypeError. Verify forward + optimizer build work."""
    import cml.nn as nn
    import cml.optim as optim

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))  # *modules ctor
    x = cml.tensor(np.random.randn(6, 4).astype(np.float32))
    out = model(x)
    assert out.shape == (6, 2)
    opt = optim.Adam(model, lr=1e-3)   # must construct without a type error
    assert opt is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
