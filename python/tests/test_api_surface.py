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
    assert np.allclose(_t(v).diag().numpy(), np.diag(v))


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
    Ap = np.random.rand(3, 4).astype(np.float32) + 1
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


def test_scatter():
    base = np.zeros((3, 4), dtype=np.float32)
    idx = np.array([[0, 1, 2, 0]], dtype=np.float32)
    src = np.array([[10, 20, 30, 40]], dtype=np.float32)
    out = _t(base).scatter(0, _t(idx), _t(src)).numpy()
    ref = base.copy()
    for j in range(4):
        ref[int(idx[0, j]), j] = src[0, j]
    assert np.allclose(out, ref)


def test_fft_forward():
    x = np.random.rand(8).astype(np.float32)
    xi = np.stack([x, np.zeros_like(x)], axis=1)
    out = _t(xi).fft().numpy()
    ref = np.fft.fft(x)
    assert np.allclose(out[:, 0], ref.real, atol=1e-3)
    assert np.allclose(out[:, 1], ref.imag, atol=1e-3)


def test_fft_inverse_and_npot():
    c = (np.random.rand(8) + 1j * np.random.rand(8)).astype(np.complex64)
    ci = np.ascontiguousarray(np.stack([c.real, c.imag], axis=1).astype(np.float32))
    inv = _t(ci).ifft().numpy()
    assert np.allclose(inv[:, 0] + 1j * inv[:, 1], np.fft.ifft(c), atol=1e-3)
    y = np.random.rand(6).astype(np.float32)  # non-power-of-two -> naive DFT path
    yi = np.ascontiguousarray(np.stack([y, np.zeros_like(y)], axis=1))
    out = _t(yi).fft().numpy()
    assert np.allclose(out[:, 0] + 1j * out[:, 1], np.fft.fft(y), atol=1e-3)


def test_fft2_and_inverse():
    img = (np.random.rand(4, 4) + 1j * np.random.rand(4, 4)).astype(np.complex64)
    ii = np.ascontiguousarray(np.stack([img.real, img.imag], axis=2).astype(np.float32))
    t = _t(ii)
    fwd = t.fft2().numpy()
    assert np.allclose(fwd[:, :, 0] + 1j * fwd[:, :, 1], np.fft.fft2(img), atol=1e-3)
    inv = _t(np.ascontiguousarray(fwd)).ifft2().numpy()
    assert np.allclose(inv[:, :, 0] + 1j * inv[:, :, 1], img, atol=1e-3)


def test_log10_lazy():
    A = np.random.rand(3, 4).astype(np.float32) * 100 + 0.01
    assert np.allclose(_t(A).log10().numpy(), np.log10(A), atol=1e-4)


def test_inplace_ops():
    for cml_fn, np_fn in [("add_", lambda a, b: a + b),
                          ("sub_", lambda a, b: a - b),
                          ("mul_", lambda a, b: a * b),
                          ("div_", lambda a, b: a / b)]:
        a = np.array([4., 9., 12.], dtype=np.float32)
        b = np.array([2., 3., 4.], dtype=np.float32)
        ta = _t(a)
        ret = getattr(ta, cml_fn)(_t(b))
        ref = np_fn(a, b)
        assert np.allclose(ret.numpy().ravel(), ref), cml_fn
        assert np.allclose(ta.numpy().ravel(), ref), f"{cml_fn} did not mutate self"


def test_any_all():
    B = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    b = _t(B)
    assert np.allclose(b.any(1).numpy().ravel(), (B != 0).any(axis=1))
    assert np.allclose(b.all(1).numpy().ravel(), (B != 0).all(axis=1))
    assert np.allclose(b.any().numpy().ravel(), (B != 0).any())
    assert np.allclose(b.all().numpy().ravel(), (B != 0).all())


def test_logsumexp():
    A = np.random.rand(4, 5).astype(np.float32)
    a = _t(A)
    assert np.allclose(a.logsumexp(1).numpy().ravel(),
                       np.log(np.sum(np.exp(A), axis=1)), atol=1e-4)


def test_unflatten():
    V = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = _t(V).unflatten(1, [2, 2]).numpy()
    assert out.shape == (3, 2, 2)
    assert np.allclose(out.reshape(3, 4), V)


def test_scatter_add_segment_sum():
    idx = np.array([0, 1, 0, 2], dtype=np.float32)
    src = np.array([1., 2., 3., 4.], dtype=np.float32)
    out = cml.scatter_add(_t(idx), _t(src), 0, 3).numpy().ravel()
    ref = np.zeros(3, dtype=np.float32)
    np.add.at(ref, idx.astype(int), src)
    assert np.allclose(out, ref)


def test_stepped_slice_correct():
    A = np.arange(24, dtype=np.float32).reshape(4, 6)
    a = _t(A)
    assert np.allclose(a[0:4:2, 0:6:2].numpy(), A[0:4:2, 0:6:2])
    assert np.allclose(a[::2].numpy(), A[::2])
    assert np.allclose(a[:, 1::3].numpy(), A[:, 1::3])


def test_comparisons_stay_lazy_and_match_numpy():
    import operator as op
    rs = np.random.RandomState(0)
    A = rs.randn(3, 4).astype(np.float32)
    B = rs.randn(3, 4).astype(np.float32)
    R = rs.randn(4).astype(np.float32)
    a, b, r = _t(A), _t(B), _t(R)
    for o in (op.lt, op.gt, op.le, op.ge, op.ne):
        for x, y, X, Y in ((a, b, A, B), (a, r, A, R), (a, 0.25, A, 0.25)):
            res = o(x, y)
            assert not res._tensor.is_executed, f"{o.__name__} realized eagerly"
            got = res.numpy()
            assert got.dtype == np.float32
            assert np.array_equal(got, o(X, Y).astype(np.float32)), o.__name__
    assert np.allclose(cml.where(a < b, a, b).numpy(), np.minimum(A, B))


def test_bool_tensor_numpy_readback():
    from cml.core import lib
    A = np.array([1, 5, 3, -2], dtype=np.float32)
    B = np.array([2, 4, 3, -1], dtype=np.float32)
    a, b = _t(A), _t(B)
    mask = cml.Tensor(lib.uop_cmplt(a._tensor, b._tensor))
    got = mask.numpy()
    assert got.dtype == np.bool_
    assert np.array_equal(got, A < B)


_SLICES = [np.s_[::2], np.s_[:, 1::3], np.s_[1:4:2, 2:6], np.s_[::3, 1::2, ::2],
           np.s_[1::2, ..., ::4], np.s_[0:4:5], np.s_[3:1], np.s_[2, ::2], np.s_[::-1],
           np.s_[1:3, ::-2], np.s_[::-3, 1::2, ::-1], np.s_[-1:-4:-1]]


@pytest.mark.parametrize("idx", _SLICES, ids=[str(s) for s in _SLICES])
def test_stepped_slicing_lazy_values_and_grad(idx):
    A = np.arange(4 * 6 * 5, dtype=np.float32).reshape(4, 6, 5)
    r = _t(A)[idx]
    assert r.numel == 0 or not r._tensor.is_executed, "slice realized eagerly"
    got = r.numpy()
    assert got.shape == A[idx].shape and np.array_equal(got, A[idx])
    if A[idx].size:
        x = _t(A.copy())
        x.requires_grad_(True)
        x[idx].sum().backward()
        want = np.zeros_like(A)
        want[idx] = 1.0
        assert np.array_equal(x.grad.numpy().reshape(A.shape), want)
    cml.reset_graph()


def test_repeated_chained_slices_do_not_corrupt_base():
    A = np.arange(4 * 6 * 5, dtype=np.float32).reshape(4, 6, 5)
    a = _t(A)
    idx = np.s_[::3, 1::2, ::2]
    first, second = a[idx].numpy(), a[idx].numpy()
    assert np.array_equal(first, A[idx]) and np.array_equal(second, A[idx])
    assert np.array_equal(a.numpy(), A)


def test_more_activations_and_lerp_vs_torch():
    import torch
    import torch.nn.functional as F
    X = (np.random.RandomState(1).randn(3, 8) * 2).astype(np.float32)
    x, T = _t(X), torch.from_numpy(X)
    for got, ref in [(x.silu(), F.silu(T)), (x.mish(), F.mish(T)), (x.selu(), F.selu(T)),
                     (x.hardswish(), F.hardswish(T)), (x.elu(0.5), F.elu(T, 0.5)),
                     (x.leaky_relu(0.1), F.leaky_relu(T, 0.1))]:
        assert np.allclose(got.numpy(), ref.numpy(), atol=1e-4)
    Y = np.random.RandomState(2).randn(3, 8).astype(np.float32)
    assert np.allclose(x.lerp(_t(Y), 0.25).numpy(), torch.lerp(T, torch.from_numpy(Y), 0.25).numpy())


@pytest.mark.parametrize("shp,ishp,dim", [((5,), (6,), 0), ((3, 4), (2, 4), 0), ((3, 4), (3, 2), 1),
                                          ((2, 3, 4), (2, 2, 3), 2), ((3, 2, 4), (2, 2, 4), 0)])
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_vs_torch(shp, ishp, dim, reduce):
    import torch
    rs = np.random.RandomState(0)
    B = rs.rand(*shp).astype(np.float32) + 0.5
    I = rs.randint(0, shp[dim], size=ishp).astype(np.float32)
    S = rs.rand(*ishp).astype(np.float32) + 0.5
    ref = torch.from_numpy(B).scatter_reduce(dim, torch.from_numpy(I).long(), torch.from_numpy(S),
                                             reduce=reduce, include_self=True)
    got = _t(B).scatter_reduce(dim, _t(I), _t(S), reduce)
    assert np.allclose(got.numpy(), ref.numpy(), atol=1e-5)


@pytest.mark.parametrize("dim,size,step", [(-1, 2, 1), (2, 3, 2), (1, 3, 2), (0, 1, 1)])
def test_unfold_vs_torch(dim, size, step):
    import torch
    A = np.arange(2 * 7 * 5, dtype=np.float32).reshape(2, 7, 5)
    assert np.array_equal(_t(A).unfold(dim, size, step).numpy(),
                          torch.from_numpy(A).unfold(dim, size, step).numpy())


def test_dtype_shorthands():
    Z = np.array([[0.0, 1.5, -2.7], [3.2, 0.0, -0.4]], dtype=np.float32)
    z = _t(Z)
    for name, np_dt in [("float", np.float32), ("double", np.float64), ("half", np.float16),
                        ("int", np.int32), ("long", np.int64), ("bool", np.bool_)]:
        got = getattr(z, name)().numpy()
        assert got.dtype == np_dt
        assert np.array_equal(got, Z.astype(np_dt)), name
    N = np.array([np.nan, 0.0, -0.0, 1e-30], dtype=np.float32)
    assert np.array_equal(_t(N).bool().numpy(), N.astype(bool))
