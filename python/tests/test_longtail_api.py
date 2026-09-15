"""Tests for the 2026-08-26 long-tail API work: fancy indexing, pad,
advanced reductions, and dtype promotion."""

import numpy as np
import pytest

import cml
from cml import Tensor, pad, result_type
from cml.core import (
    DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT32,
)


def make(x):
    return Tensor(np.ascontiguousarray(x, dtype=np.float32))


class TestIndexing:
    def test_slices_2d(self):
        rng = np.random.RandomState(0)
        x = rng.rand(4, 5).astype(np.float32)
        t = make(x)
        assert np.allclose(t[1:3].numpy(), x[1:3])
        assert np.allclose(t[:, 2].numpy(), x[:, 2])
        assert np.allclose(t[::2].numpy(), x[::2])          # step -> numpy path
        assert np.allclose(t[-1].numpy(), x[-1])

    def test_slice_is_lazy(self):
        """Slicing an unrealized graph output must not force full-graph eager
        evaluation of unrelated nodes (regression: view-of-lazy null data)."""
        t = make([[1.0, 2.0], [3.0, 4.0]])
        out = t.sum(0).reshape(2)
        assert np.allclose(out.numpy(), [4.0, 6.0])

    def test_fancy_index(self):
        rng = np.random.RandomState(1)
        x = rng.rand(4, 5).astype(np.float32)
        t = make(x)
        assert np.allclose(t[[0, 2]].numpy(), x[[0, 2]])
        assert np.allclose(t[np.array([3, 1])].numpy(), x[[3, 1]])
        assert np.allclose(t[Tensor([[0.0, 2.0]])].numpy(), x[np.array([[0, 2]])])
        assert np.allclose(t[:, [1, 3]].numpy(), x[:, [1, 3]])

    def test_fancy_index_inner_dim(self):
        rng = np.random.RandomState(2)
        x = rng.rand(3, 4, 5).astype(np.float32)
        t = make(x)
        assert np.allclose(t[:, [1, 3], :].numpy(), x[:, [1, 3], :])
        assert np.allclose(t[..., 2].numpy(), x[..., 2])
        assert np.allclose(t[:, :, [0, 2, 4]].numpy(), x[:, :, [0, 2, 4]])

    def test_bool_mask(self):
        rng = np.random.RandomState(3)
        x = rng.rand(4, 4).astype(np.float32)
        t = make(x)
        m = x > 0.5
        assert np.allclose(t[m].numpy(), x[m])              # ndarray mask
        assert np.allclose(t[(t > 0.5)].numpy(), x[x > 0.5])  # tensor mask

    def test_negative_and_bounds(self):
        t = make([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(IndexError):
            t[5]
        with pytest.raises(IndexError):
            t[[0, 9]]
        assert np.allclose(t[-1].numpy(), [3.0, 4.0])


class TestPad:
    def test_torch_order_flat(self):
        x = make([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(cml.pad(x, (1, 1)).numpy(),
                           np.pad(x.numpy(), ((0, 0), (1, 1))))
        assert np.allclose(x.pad((1, 1, 1, 1)).numpy(),
                           np.pad(x.numpy(), ((1, 1), (1, 1))))
        # fewer pairs pad only trailing dims (torch F.pad semantics)
        y = make(np.arange(6, dtype=np.float32).reshape(1, 2, 3))
        assert np.allclose(y.pad((1, 1)).numpy(),
                           np.pad(y.numpy(), ((0, 0), (0, 0), (1, 1))))

    def test_nested_dim_order(self):
        x = make([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(x.pad(((1, 1), (2, 2))).numpy(),
                           np.pad(x.numpy(), ((1, 1), (2, 2))))

    def test_int_all_dims(self):
        x = make([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(x.pad(1).numpy(), np.pad(x.numpy(), 1))

    def test_reflect_replicate(self):
        x = make(np.arange(12, dtype=np.float32).reshape(3, 4))
        r = x.pad(((1, 1), (0, 0)), mode="reflect").numpy()
        assert r.shape == (5, 4)
        assert np.allclose(r[0], r[2])                      # reflect rows
        p = x.pad(((1, 1), (0, 0)), mode="edge").numpy()
        assert np.allclose(p[0], p[1])                      # replicated row

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            make([[1.0]]).pad((1, 1), mode="bogus")


class TestReductions:
    def test_cumsum_cumprod_argsort(self):
        rng = np.random.RandomState(4)
        x = rng.rand(3, 4).astype(np.float32)
        t = make(x)
        assert np.allclose(t.cumsum(-1).numpy(), np.cumsum(x, -1))
        assert np.allclose(t.cumprod(-1).numpy(), np.cumprod(x, -1))
        assert np.allclose(t.argsort(1).numpy(), np.argsort(x, 1))

    def test_topk_values_indices(self):
        rng = np.random.RandomState(5)
        x = rng.rand(4, 6).astype(np.float32)
        vals, idx = make(x).topk(3, dim=1)
        assert np.allclose(vals.numpy(), np.sort(x, 1)[:, -3:][:, ::-1])
        assert np.allclose(idx.numpy(), np.argsort(-x, 1)[:, :3])

    def test_median(self):
        rng = np.random.RandomState(6)
        x = rng.rand(4, 7).astype(np.float32)
        t = make(x)
        assert abs(t.median().item() - float(np.median(x))) < 1e-5
        assert np.allclose(t.median(dim=1).numpy(), np.median(x, 1))

    def test_unique_masked_select(self):
        u = make([3.0, 1.0, 3.0, 2.0])
        assert np.allclose(u.unique().numpy(), [1.0, 2.0, 3.0])
        x = make([[1.0, 5.0], [2.0, 6.0]])
        assert np.allclose(x.masked_select(x > 2.5).numpy(), [5.0, 6.0])


class TestPromotion:
    def test_result_type(self):
        f32 = make([[1.0]])
        f64 = f32.cast(DTYPE_FLOAT64)
        i32 = f32.cast(DTYPE_INT32)
        assert result_type(f32, f64) == DTYPE_FLOAT64
        assert result_type(i32, f32) == DTYPE_FLOAT32
        assert result_type(i32, i32) == DTYPE_INT32

    def test_tensor_tensor_promotion(self):
        f32 = make([[1.0]])
        f64 = f32.cast(DTYPE_FLOAT64)
        i32 = f32.cast(DTYPE_INT32)
        assert (f64 + i32).dtype == DTYPE_FLOAT64
        assert (f32 * f64).dtype == DTYPE_FLOAT64
        assert np.allclose((f64 + i32).numpy(), 2.0)

    def test_weak_scalars(self):
        f64 = make([[1.0]]).cast(DTYPE_FLOAT64)
        i32 = make([[1.0]]).cast(DTYPE_INT32)
        assert (i32 + 1).dtype == DTYPE_INT32               # int stays int
        assert (1 + i32).dtype == DTYPE_INT32
        assert (i32 + 0.5).dtype == DTYPE_FLOAT32           # float widens int
        assert np.allclose((i32 + 0.5).numpy(), 1.5)
        assert (f64 - 1).dtype == DTYPE_FLOAT64             # int never widens f64
        assert (2 - f64).dtype == DTYPE_FLOAT64

    def test_values_across_ops(self):
        a = make([1.0, 2.0, 3.0])
        b = make([10.0, 20.0, 30.0])
        assert np.allclose((a + b * 2 - 1).numpy(),
                           np.asarray([20.0, 41.0, 62.0], dtype=np.float32))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
