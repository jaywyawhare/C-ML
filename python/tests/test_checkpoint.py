"""Tests for cml.checkpoint (activation rematerialization helper)."""

import numpy as np
import pytest

import cml
import cml.nn as cnn
from cml.checkpoint import checkpoint


def _make(seed=3):
    np.random.seed(seed)
    l1, l2 = cnn.Linear(8, 8), cnn.Linear(8, 4)
    X = np.random.randn(4, 8).astype(np.float32)
    return l1, l2, X


def test_checkpoint_matches_eager():
    l1, l2, X = _make()
    h = l1(cml.Tensor(X.copy()))
    out_ckpt = checkpoint(lambda hh: l2(hh), h)

    cml.reset_graph()
    h2 = l1(cml.Tensor(X.copy()))
    out_eager = l2(h2)

    assert out_ckpt.numpy().shape == (4, 4)
    assert np.allclose(out_ckpt.numpy(), out_eager.numpy(), atol=1e-5)


def test_checkpoint_multiple_outputs():
    l1, _, X = _make()
    h = l1(cml.Tensor(X.copy()))
    a, b = checkpoint(lambda hh: (hh * 2.0, hh + 1.0), h)
    cml.reset_graph()
    h2 = l1(cml.Tensor(X.copy()))
    assert np.allclose(a.numpy(), h2.numpy() * 2.0, atol=1e-5)
    assert np.allclose(b.numpy(), h2.numpy() + 1.0, atol=1e-5)


def test_checkpoint_determinism_flag():
    l1, _, X = _make()
    h = l1(cml.Tensor(X.copy()))
    # deterministic segment passes the check
    checkpoint(lambda hh: hh + 1.0, h, verify_determinism=True)


def test_checkpoint_flags_nondeterministic_segment():
    def rand_seg(h):
        noise = cml.Tensor(
            np.random.randn(*h.numpy().shape).astype(np.float32)
        )
        return h + noise

    h = cml.Tensor(np.ones((3, 3), dtype=np.float32))
    with pytest.raises(RuntimeError):
        checkpoint(rand_seg, h, verify_determinism=True)
