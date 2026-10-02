"""Step-for-step convergence parity against PyTorch.

Trains the same tiny MLP in C-ML and torch with identical initial weights,
identical batches, plain SGD and MSE, and requires the loss curves to agree
per step. Skips cleanly when torch is not installed.

Run:  cd python && python3 -m pytest tests/test_torch_parity.py -q -s
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
import cml.nn as cnn
import cml.losses as closses
import cml.optim as coptim
from cml.safetensors import _copy_into

cml.init()


def _torch_from(t_np):
    return torch.tensor(np.array(t_np, dtype=np.float32))


def build_pair(seed=0, in_f=4, hidden=8, out_f=2):
    """Build both models; C-ML owns the initialization, torch mirrors it."""
    del seed
    m_cml = cnn.Sequential(
        cnn.Linear(in_f, hidden),
        cnn.ReLU(),
        cnn.Linear(hidden, out_f),
    )

    # read cml's initial weights ([out,in] layout, like torch.nn.Linear)
    w1 = np.asarray(m_cml[0].parameters()[0].tensor.numpy())
    b1 = np.asarray(m_cml[0].parameters()[1].tensor.numpy())
    w2 = np.asarray(m_cml[2].parameters()[0].tensor.numpy())
    b2 = np.asarray(m_cml[2].parameters()[1].tensor.numpy())

    m_torch = torch.nn.Sequential(
        torch.nn.Linear(in_f, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, out_f),
    )
    with torch.no_grad():
        m_torch[0].weight.copy_(_torch_from(w1))
        m_torch[0].bias.copy_(_torch_from(b1))
        m_torch[2].weight.copy_(_torch_from(w2))
        m_torch[2].bias.copy_(_torch_from(b2))

    return m_cml, m_torch


def test_forward_parity():
    m_cml, m_torch = build_pair()
    x = np.random.RandomState(1).randn(5, 4).astype(np.float32)

    with torch.no_grad():
        y_torch = m_torch(_torch_from(x)).numpy()

    y_cml = np.asarray(m_cml(cml.Tensor(x.copy())).numpy())
    assert y_cml.shape == y_torch.shape
    assert np.allclose(y_cml, y_torch, rtol=1e-4, atol=1e-5), (
        f"forward mismatch: max|d|={np.abs(y_cml - y_torch).max():.3e}")


def test_training_loss_curve_parity():
    m_cml, m_torch = build_pair(seed=7)

    rs = np.random.RandomState(42)
    X = rs.randn(32, 4).astype(np.float32)
    Y = (X[:, :2] * 1.5 + 0.25).astype(np.float32)

    x_t = _torch_from(X)
    y_t = _torch_from(Y)

    opt_t = torch.optim.SGD(m_torch.parameters(), lr=0.05)
    crit_t = torch.nn.MSELoss()

    opt_c = coptim.SGD(m_cml, lr=0.05)
    losses_cml, losses_torch = [], []

    for step in range(20):
        # torch step
        opt_t.zero_grad()
        pred_t = m_torch(x_t)
        loss_t = crit_t(pred_t, y_t)
        loss_t.backward()
        opt_t.step()
        losses_torch.append(float(loss_t.item()))

        # cml step — same lifecycle as the C training loop: realize inputs,
        # build a fresh graph per step (reset after the update), because
        # backward #N re-executes every prior step's VJP nodes otherwise
        pred_c = m_cml(cml.Tensor(X.copy()))
        loss_c = closses.mse_loss(pred_c, cml.Tensor(Y.copy()))
        lv = float(np.asarray(loss_c.numpy()).ravel()[0])
        loss_c.backward()
        opt_c.step()
        opt_c.zero_grad()
        cml.reset_graph()
        losses_cml.append(lv)

    diffs = np.abs(np.array(losses_cml) - np.array(losses_torch))
    print(f"\nmax |loss diff| over 20 steps: {diffs.max():.3e}")
    print(f"cml   head: {['%.4f' % v for v in losses_cml[:5]]}")
    print(f"torch head: {['%.4f' % v for v in losses_torch[:5]]}")
    assert diffs.max() < 1e-3, f"loss curves diverge (max {diffs.max():.3e})"


def test_cnn_training_loss_curve_parity():
    """Conv -> ReLU -> flatten -> Linear, trained step-for-step vs torch.

    Proves conv forward AND its gradient path (the heaviest VJP family)
    match torch numerically over multiple optimizer steps."""
    m_cml = cnn.Sequential(
        cnn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        cnn.ReLU(),
        cnn.Flatten(),
        cnn.Linear(4 * 8 * 8, 2),
    )

    # mirror weights into torch: read cml's init, copy across
    w_conv = np.asarray(m_cml[0].parameters()[0].tensor.numpy())
    b_conv = np.asarray(m_cml[0].parameters()[1].tensor.numpy())
    w_fc = np.asarray(m_cml[3].parameters()[0].tensor.numpy())
    b_fc = np.asarray(m_cml[3].parameters()[1].tensor.numpy())

    m_torch = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 8 * 8, 2),
    )
    with torch.no_grad():
        m_torch[0].weight.copy_(_torch_from(w_conv))
        m_torch[0].bias.copy_(_torch_from(b_conv))
        m_torch[3].weight.copy_(_torch_from(w_fc))
        m_torch[3].bias.copy_(_torch_from(b_fc))

    rs = np.random.RandomState(7)
    X = rs.randn(8, 1, 8, 8).astype(np.float32)
    Y = rs.randn(8, 2).astype(np.float32)

    x_t = _torch_from(X)
    y_t = _torch_from(Y)

    opt_t = torch.optim.SGD(m_torch.parameters(), lr=0.03)
    crit_t = torch.nn.MSELoss()

    opt_c = coptim.SGD(m_cml, lr=0.03)
    losses_cml, losses_torch = [], []

    for step in range(10):
        opt_t.zero_grad()
        loss_t = crit_t(m_torch(x_t), y_t)
        loss_t.backward()
        opt_t.step()
        losses_torch.append(float(loss_t.item()))

        pred_c = m_cml(cml.Tensor(X.copy()))
        loss_c = closses.mse_loss(pred_c, cml.Tensor(Y.copy()))
        lv = float(np.asarray(loss_c.numpy()).ravel()[0])
        loss_c.backward()
        opt_c.step()
        opt_c.zero_grad()
        cml.reset_graph()
        losses_cml.append(lv)

    diffs = np.abs(np.array(losses_cml) - np.array(losses_torch))
    print(f"\nmax |cnn loss diff| over 10 steps: {diffs.max():.3e}")
    assert diffs.max() < 5e-3, f"CNN loss curves diverge (max {diffs.max():.3e})"


def test_transformer_block_training_loss_curve_parity():
    """Single-head self-attention block trained step-for-step vs torch.

    Attention is composed from primitives (linear projections, matmul,
    scaled softmax) on BOTH sides with identical weights, so this pins the
    full composition path — including softmax and per-step gradient flow
    through attention — not just individual ops."""
    d, seq, batch = 8, 6, 4

    def mirror(torch_mod, cml_lin):
        """copy a torch Linear's weights into a cml Linear"""
        w = np.asarray(cml_lin.parameters()[0].tensor.numpy())
        b = np.asarray(cml_lin.parameters()[1].tensor.numpy())
        with torch.no_grad():
            torch_mod.weight.copy_(_torch_from(w))
            torch_mod.bias.copy_(_torch_from(b))

    # --- cml side ---------------------------------------------------------
    lq = cnn.Linear(d, d); lk = cnn.Linear(d, d); lv = cnn.Linear(d, d)
    lo = cnn.Linear(d, d)
    # container exists only so the optimizer can collect/step the parameters;
    # the attention math below wires the layers manually.
    block_cml = cnn.Sequential(lq, lk, lv, lo)

    # --- torch side -------------------------------------------------------
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = torch.nn.Linear(d, d)
            self.wk = torch.nn.Linear(d, d)
            self.wv = torch.nn.Linear(d, d)
            self.wo = torch.nn.Linear(d, d)

        def forward(self, x):
            q = self.wq(x); k = self.wk(x); v = self.wv(x)
            scores = q @ k.transpose(-2, -1) / (d ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            return self.wo(attn @ v)

    m_torch = Block()
    mirror(m_torch.wq, lq); mirror(m_torch.wk, lk)
    mirror(m_torch.wv, lv); mirror(m_torch.wo, lo)

    rs = np.random.RandomState(11)
    X = rs.randn(batch, seq, d).astype(np.float32)
    Y = rs.randn(batch, seq, d).astype(np.float32)
    x_t = _torch_from(X); y_t = _torch_from(Y)

    opt_t = torch.optim.SGD(m_torch.parameters(), lr=0.02)
    crit_t = torch.nn.MSELoss()
    opt_c = coptim.SGD(block_cml, lr=0.02)

    scale = float(d) ** 0.5
    losses_cml, losses_torch = [], []

    def flatten(x):
        if x.ndim == 3:
            return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        return x

    def unflatten(x, b, s, dd):
        return x.reshape(b, s, dd)

    for step in range(8):
        opt_t.zero_grad()
        loss_t = crit_t(m_torch(x_t), y_t)
        loss_t.backward()
        opt_t.step()
        losses_torch.append(float(loss_t.item()))

        x = cml.Tensor(X.copy())
        # project in 2-D, attend in 3-D (batched matmul keeps [B,S,*])
        q = unflatten(lq(flatten(x)), batch, seq, d)
        k = unflatten(lk(flatten(x)), batch, seq, d)
        v = unflatten(lv(flatten(x)), batch, seq, d)
        scores = q.matmul(k.transpose(1, 2)) / scale
        attn = scores.softmax(-1)
        ctx = attn.matmul(v)
        out = lo(flatten(ctx))

        loss_c = closses.mse_loss(unflatten(out, batch, seq, d), cml.Tensor(Y.copy()))
        loss_val = float(np.asarray(loss_c.numpy()).ravel()[0])
        loss_c.backward()
        opt_c.step()
        opt_c.zero_grad()
        cml.reset_graph()
        losses_cml.append(loss_val)

    diffs = np.abs(np.array(losses_cml) - np.array(losses_torch))
    print(f"\nmax |transformer loss diff| over 8 steps: {diffs.max():.3e}")
    assert all(np.isfinite(losses_cml)) and losses_cml[-1] < losses_cml[0], \
        "cml transformer block does not converge"
    assert diffs.max() < 1e-4, \
        f"cml transformer loss curve diverges from torch (max {diffs.max():.3e})"


# ── optimizer parity ───────────────────────────────────────────────────────
# Push identical known weights into a C-ML and a torch Linear, train both on
# the same batch, and require the *parameters* (not just the loss) to agree
# after every step. This is what catches an optimizer-math divergence.
def _synced_linear(in_f=4, out_f=3, seed=0):
    rs = np.random.RandomState(seed)
    W = rs.randn(out_f, in_f).astype(np.float32)
    B = rs.randn(out_f).astype(np.float32)
    lin_c = cnn.Linear(in_f, out_f)
    model_c = cnn.Sequential(lin_c)
    pc = lin_c.parameters()
    _copy_into(pc[0].tensor, W)
    _copy_into(pc[1].tensor, B)
    lin_t = torch.nn.Linear(in_f, out_f)
    with torch.no_grad():
        lin_t.weight.copy_(_torch_from(W))
        lin_t.bias.copy_(_torch_from(B))
    return model_c, pc, lin_t


_OPTIMIZERS = [
    ("sgd",        lambda m: coptim.SGD(m, lr=0.1),
                   lambda l: torch.optim.SGD(l.parameters(), lr=0.1)),
    ("sgd_mom",    lambda m: coptim.SGD(m, lr=0.1, momentum=0.9),
                   lambda l: torch.optim.SGD(l.parameters(), lr=0.1, momentum=0.9)),
    ("sgd_mom_wd", lambda m: coptim.SGD(m, lr=0.1, momentum=0.9, weight_decay=0.01),
                   lambda l: torch.optim.SGD(l.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)),
    ("adam",       lambda m: coptim.Adam(m, lr=0.05),
                   lambda l: torch.optim.Adam(l.parameters(), lr=0.05)),
    ("adamw",      lambda m: coptim.AdamW(m, lr=0.05, weight_decay=0.01),
                   lambda l: torch.optim.AdamW(l.parameters(), lr=0.05, weight_decay=0.01)),
    ("rmsprop",    lambda m: coptim.RMSprop(m, lr=0.05),
                   lambda l: torch.optim.RMSprop(l.parameters(), lr=0.05)),
    ("adagrad",    lambda m: coptim.AdaGrad(m, lr=0.1),
                   lambda l: torch.optim.Adagrad(l.parameters(), lr=0.1)),
]


@pytest.mark.parametrize("name,make_c,make_t", _OPTIMIZERS)
def test_optimizer_param_parity(name, make_c, make_t):
    model_c, pc, lin_t = _synced_linear(seed=0)
    oc = make_c(model_c)
    ot = make_t(lin_t)
    rs = np.random.RandomState(7)
    X = rs.randn(16, 4).astype(np.float32)
    Y = rs.randn(16, 3).astype(np.float32)
    for step in range(5):
        pred = model_c(cml.Tensor(X.copy()))
        loss = closses.mse_loss(pred, cml.Tensor(Y.copy()))
        loss.backward()
        oc.step()
        oc.zero_grad()
        cml.reset_graph()

        ot.zero_grad()
        F.mse_loss(lin_t(_torch_from(X)), _torch_from(Y)).backward()
        ot.step()

        wc = np.asarray(pc[0].tensor.numpy())
        bc = np.asarray(pc[1].tensor.numpy())
        assert np.allclose(wc, lin_t.weight.detach().numpy(), atol=2e-4), f"{name} weight @step{step}"
        assert np.allclose(bc, lin_t.bias.detach().numpy(), atol=2e-4), f"{name} bias @step{step}"


# ── loss parity (forward + input-grad) ─────────────────────────────────────
def _loss_fwd_grad(x_np, cml_loss, torch_loss):
    xc = cml.Tensor(x_np.copy())
    xc.requires_grad_(True)
    lc = cml_loss(xc)
    fv = float(np.asarray(lc.numpy()).ravel()[0])
    lc.backward()
    gc = np.asarray(xc.grad.numpy()).reshape(x_np.shape)
    xt = torch.tensor(x_np, requires_grad=True)
    lt = torch_loss(xt)
    ft = float(lt.item())
    lt.backward()
    return fv, ft, gc, xt.grad.numpy()


def test_loss_parity():
    rs = np.random.RandomState(1)
    P = rs.randn(8, 5).astype(np.float32)
    T = rs.randn(8, 5).astype(np.float32)
    lab = rs.randint(0, 5, size=8)
    labf = lab.astype(np.float32)
    prob = (rs.rand(8, 5) * 0.98 + 0.01).astype(np.float32)
    bt = (rs.rand(8, 5) > 0.5).astype(np.float32)
    cases = [
        ("mse", P, lambda t: closses.mse_loss(t, cml.Tensor(T.copy())),
                   lambda t: F.mse_loss(t, _torch_from(T))),
        ("mae", P, lambda t: closses.mae_loss(t, cml.Tensor(T.copy())),
                   lambda t: F.l1_loss(t, _torch_from(T))),
        ("cross_entropy", rs.randn(8, 5).astype(np.float32),
                   lambda t: closses.cross_entropy_loss(t, cml.Tensor(labf.copy())),
                   lambda t: F.cross_entropy(t, torch.tensor(lab))),
        ("bce_prob", prob, lambda t: closses.bce_loss(t, cml.Tensor(bt.copy())),
                   lambda t: F.binary_cross_entropy(t, _torch_from(bt))),
    ]
    for name, x, cf, tf in cases:
        fv, ft, gc, gt = _loss_fwd_grad(x, cf, tf)
        assert abs(fv - ft) < 2e-4, f"{name} forward {fv:.5f} vs {ft:.5f}"
        assert np.allclose(gc, gt, atol=2e-4, rtol=2e-4), f"{name} input-grad"


def test_nll_parity():
    rs = np.random.RandomState(2)
    lab = rs.randint(0, 5, size=8)
    labf = lab.astype(np.float32)
    logp = np.log(F.softmax(torch.tensor(rs.randn(8, 5).astype(np.float32)), dim=1).numpy()
                  + 1e-9).astype(np.float32)
    fv, ft, gc, gt = _loss_fwd_grad(
        logp,
        lambda t: closses.nll_loss(t, cml.Tensor(labf.copy())),
        lambda t: F.nll_loss(t, torch.tensor(lab)))
    assert abs(fv - ft) < 2e-4
    assert np.allclose(gc, gt, atol=2e-4, rtol=2e-4)


# huber: forward + backward, over input ranges that exercise BOTH the quadratic
# (|err|<=delta) and linear regions. The quadratic-region backward was the bug
# (where/cmplt branch-select never picked the quadratic branch); now clamp-based.
@pytest.mark.parametrize("scale,delta", [(0.2, 1.0), (1.0, 1.0), (3.0, 1.0),
                                         (1.0, 0.5), (2.0, 2.0)])
def test_huber_parity(scale, delta):
    rs = np.random.RandomState(3)
    P = (rs.rand(10, 4).astype(np.float32) * scale)
    T = (rs.rand(10, 4).astype(np.float32) * scale)
    fv, ft, gc, gt = _loss_fwd_grad(
        P,
        lambda t: closses.huber_loss(t, cml.Tensor(T.copy()), delta),
        lambda t: F.huber_loss(t, _torch_from(T), delta=delta))
    assert abs(fv - ft) < 2e-4, f"huber forward (scale={scale}, delta={delta})"
    assert np.allclose(gc, gt, atol=2e-4, rtol=2e-4), \
        f"huber backward (scale={scale}, delta={delta})"
