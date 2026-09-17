"""NN-layer and shape-op forward+backward parity against PyTorch.

Layers (LayerNorm, GroupNorm, Conv2d, Embedding) are the highest-value coverage:
they compose the reductions/broadcasts/matmuls that transformers and CNNs rely
on, and mismatches there break whole architectures. C-ML owns the weight init;
torch mirrors it, then we compare forward and gradients.

Run:  cd python && PYTHONPATH=. python -m pytest tests/test_nn_parity.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CML_LOG_LEVEL", "ERROR")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import cml
import cml.nn as cnn

cml.init()
TOL = 3e-4
rs = np.random.RandomState(3)


def _params(layer):
    return [np.asarray(p.tensor.numpy()) for p in layer.parameters()]


def test_layernorm_parity():
    d = 8
    ln = cnn.LayerNorm(d)
    w, b = _params(ln)  # gamma, beta, shape (d,)
    tln = torch.nn.LayerNorm(d)
    with torch.no_grad():
        tln.weight.copy_(torch.tensor(w.reshape(-1)))
        tln.bias.copy_(torch.tensor(b.reshape(-1)))

    X = rs.randn(4, d).astype(np.float32)
    xc = cml.Tensor(X.copy()); xc.requires_grad_(True)
    yc = ln(xc); yc.sum().backward()
    xt = torch.tensor(X, requires_grad=True)
    yt = tln(xt); yt.sum().backward()

    assert np.allclose(np.asarray(yc.numpy()), yt.detach().numpy(), atol=TOL), \
        f"layernorm fwd (max {np.abs(np.asarray(yc.numpy()) - yt.detach().numpy()).max():.3e})"
    assert np.allclose(np.asarray(xc.grad.numpy()), xt.grad.numpy(), atol=TOL), \
        f"layernorm grad (max {np.abs(np.asarray(xc.grad.numpy()) - xt.grad.numpy()).max():.3e})"


def test_conv2d_parity():
    cin, cout, k = 3, 4, 3
    conv = cnn.Conv2d(cin, cout, k, stride=1, padding=1)
    p = _params(conv)
    w = p[0]  # (cout, cin, k, k)
    b = p[1] if len(p) > 1 else None
    tconv = torch.nn.Conv2d(cin, cout, k, stride=1, padding=1)
    with torch.no_grad():
        tconv.weight.copy_(torch.tensor(w.reshape(cout, cin, k, k)))
        if b is not None:
            tconv.bias.copy_(torch.tensor(b.reshape(-1)))

    X = rs.randn(2, cin, 8, 8).astype(np.float32)
    xc = cml.Tensor(X.copy()); xc.requires_grad_(True)
    yc = conv(xc); yc.sum().backward()
    xt = torch.tensor(X, requires_grad=True)
    yt = tconv(xt); yt.sum().backward()

    gc = np.asarray(yc.numpy()).reshape(yt.shape)
    assert np.allclose(gc, yt.detach().numpy(), atol=1e-3), \
        f"conv2d fwd (max {np.abs(gc - yt.detach().numpy()).max():.3e})"
    assert np.allclose(np.asarray(xc.grad.numpy()).reshape(X.shape), xt.grad.numpy(), atol=1e-3), \
        "conv2d input grad"


def test_groupnorm_parity():
    groups, ch = 2, 4
    gn = cnn.GroupNorm(groups, ch)
    w, b = _params(gn)
    tgn = torch.nn.GroupNorm(groups, ch)
    with torch.no_grad():
        tgn.weight.copy_(torch.tensor(w.reshape(-1)))
        tgn.bias.copy_(torch.tensor(b.reshape(-1)))
    X = rs.randn(2, ch, 5, 5).astype(np.float32)
    xc = cml.Tensor(X.copy()); xc.requires_grad_(True)
    yc = gn(xc); yc.sum().backward()
    xt = torch.tensor(X, requires_grad=True)
    yt = tgn(xt); yt.sum().backward()
    assert np.allclose(np.asarray(yc.numpy()).reshape(yt.shape), yt.detach().numpy(), atol=1e-3), \
        "groupnorm fwd"
    assert np.allclose(np.asarray(xc.grad.numpy()).reshape(X.shape), xt.grad.numpy(), atol=1e-3), \
        "groupnorm grad"


def test_batchnorm2d_training_parity():
    """Training-mode BN: the backward must flow through the batch mean/var (they
    depend on the input), not treat them as constants."""
    C = 3
    bn = cnn.BatchNorm2d(C)
    bn.train()
    w, b = _params(bn)
    X = rs.randn(4, C, 6, 6).astype(np.float32)
    W = rs.randn(4, C, 6, 6).astype(np.float32)  # shared loss weights

    xc = cml.Tensor(X.copy()); xc.requires_grad_(True)
    oc = bn(xc)
    (oc * cml.Tensor(W.copy())).sum().backward()
    gc = np.asarray(xc.grad.numpy()).reshape(X.shape)

    tbn = torch.nn.BatchNorm2d(C); tbn.train()
    with torch.no_grad():
        tbn.weight.copy_(torch.tensor(w.reshape(-1)))
        tbn.bias.copy_(torch.tensor(b.reshape(-1)))
    xt = torch.tensor(X, requires_grad=True)
    (tbn(xt) * torch.tensor(W)).sum().backward()

    assert np.allclose(np.asarray(oc.numpy()).reshape(X.shape), tbn(torch.tensor(X)).detach().numpy(),
                       atol=1e-3), "batchnorm2d fwd"
    assert np.allclose(gc, xt.grad.numpy(), atol=1e-3), \
        f"batchnorm2d training grad (max {np.abs(gc - xt.grad.numpy()).max():.3e})"


# ── shape ops ──────────────────────────────────────────────────────────────
def test_unsqueeze_squeeze_parity():
    X = rs.randn(4, 5).astype(np.float32)
    for name, cf, tf in [
        ("unsqueeze", lambda t: t.unsqueeze(1), lambda t: t.unsqueeze(1)),
        ("squeeze",   lambda t: t.unsqueeze(1).squeeze(1), lambda t: t.unsqueeze(1).squeeze(1)),
    ]:
        xc = cml.Tensor(X.copy()); xc.requires_grad_(True)
        oc = cf(xc); oc.sum().backward()
        xt = torch.tensor(X, requires_grad=True)
        ot = tf(xt); ot.sum().backward()
        assert np.allclose(np.asarray(oc.numpy()).reshape(ot.shape), ot.detach().numpy(), atol=TOL), f"{name} fwd"
        assert np.allclose(np.asarray(xc.grad.numpy()).reshape(X.shape), xt.grad.numpy(), atol=TOL), f"{name} grad"


def test_pad_parity():
    X = rs.randn(3, 4).astype(np.float32)
    xc = cml.Tensor(X.copy()); xc.requires_grad_(True)
    oc = xc.pad([1, 1, 2, 2]); oc.sum().backward()   # last-dim (1,1), then (2,2)
    xt = torch.tensor(X, requires_grad=True)
    ot = torch.nn.functional.pad(xt, [1, 1, 2, 2]); ot.sum().backward()
    assert np.allclose(np.asarray(oc.numpy()).reshape(ot.shape), ot.detach().numpy(), atol=TOL), "pad fwd"
    assert np.allclose(np.asarray(xc.grad.numpy()).reshape(X.shape), xt.grad.numpy(), atol=TOL), "pad grad"
