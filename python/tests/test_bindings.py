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


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Clear the IR graph + execution/buffer caches after each test. By teardown
    time the test's local tensors/optimizers are already collected, so this full
    reset runs at a controlled point (no GC-order fragility) and stops per-test
    state from accumulating across the suite."""
    yield
    cml.reset_graph()


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


def test_single_model_training_converges():
    """A hand-written training loop must converge — previously the autograd graph
    accumulated on the reused parameters and the loss diverged/hung. backward()
    now detaches the loss and step() resets the per-step graph automatically."""
    import cml.nn as nn
    import cml.optim as optim

    N = 20
    x = (np.arange(N) / (N - 1) * 2 - 1).astype(np.float32).reshape(N, 1)
    y = (2 * x + 1).astype(np.float32)
    X, Y = cml.tensor(x), cml.tensor(y)
    lin = nn.Linear(1, 1)
    lin.train(True)
    opt = optim.SGD(lin, lr=0.05)
    lv = None
    for _ in range(400):
        opt.zero_grad()
        loss = cml.mse_loss(lin(X), Y)
        cml.backward(loss)
        opt.step()
        lv = float(np.asarray(loss.numpy()).reshape(-1)[0])
    assert lv is not None and lv < 0.01, f"did not converge: {lv}"


def test_multiple_models_with_reset_graph():
    """Training several models in one process works when reset_graph() is called
    between them (drops the shape-keyed plan cache from the previous model)."""
    import cml.nn as nn
    import cml.optim as optim

    np.random.seed(0)
    N = 24
    x = np.random.randn(N, 3).astype(np.float32)
    y = (x @ np.array([1.0, -2.0, 0.5], np.float32) + 0.3).reshape(N, 1).astype(np.float32)
    X, Y = cml.tensor(x), cml.tensor(y)

    finals = []
    for _ in range(3):
        m = nn.Linear(3, 1)
        opt = optim.SGD(m, lr=0.1)
        lv = None
        for _ in range(300):
            opt.zero_grad()
            loss = cml.mse_loss(m(X), Y)
            cml.backward(loss)
            opt.step()
            lv = float(np.asarray(loss.numpy()).reshape(-1)[0])
        finals.append(lv)
        del m, opt, loss
        cml.reset_graph()

    assert all(f < 0.05 for f in finals), f"a model did not converge: {finals}"


def test_cross_entropy_gradient_matches_analytic():
    """cross_entropy's gradient must flow to the logits. It used to be silently
    zero (None): the gather VJP's scatter_add rejected the rank-reduced gradient,
    so no gradient reached the input. The correct gradient is softmax - onehot,
    averaged over the batch."""
    np.random.seed(0)
    logits = cml.tensor(np.random.randn(5, 4).astype(np.float32), requires_grad=True)
    labels = cml.tensor(np.array([0, 3, 1, 2, 0], np.float32))
    loss = cml.cross_entropy_loss(logits, labels)
    cml.backward(loss)
    g = cml.get_grad(logits)
    assert g is not None, "cross_entropy produced no gradient for the logits"
    raw = logits.numpy()
    sm = np.exp(raw - raw.max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)
    onehot = np.eye(4)[[0, 3, 1, 2, 0]]
    expected = (sm - onehot) / 5
    np.testing.assert_allclose(np.asarray(g.numpy()), expected, atol=1e-4)
    # This test does a bare forward+backward with no optimizer, so no step/__del__
    # reset fires to drop the accumulated graph. Clear it explicitly so it doesn't
    # contaminate the next test.
    del loss, g
    cml.reset_graph()


def test_bce_training_converges():
    """A binary-classification training loop with bce_loss must converge (a
    non-softmax loss path that trains reliably under the per-step graph reset)."""
    import cml.nn as nn
    import cml.optim as optim

    np.random.seed(0)
    N = 40
    X = np.random.randn(N, 4).astype(np.float32)
    Y = (np.random.rand(N, 1) > 0.5).astype(np.float32)
    Xt, Yt = cml.tensor(X), cml.tensor(Y)
    m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
    m.set_training(True)
    opt = optim.Adam(m, lr=0.02)
    first = last = None
    for _ in range(25):
        opt.zero_grad()
        loss = cml.bce_loss(m(Xt), Yt)
        cml.backward(loss)
        opt.step()
        last = float(np.asarray(loss.numpy()).reshape(-1)[0])
        if first is None:
            first = last
    assert last < first, f"bce did not decrease: {first} -> {last}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
