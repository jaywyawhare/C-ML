import torch


def test_reductions(opout):
    x2 = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float32)
    assert abs(opout["SUM"] - torch.sum(x2).item()) < 1e-6
    assert abs(opout["MEAN"] - torch.mean(x2).item()) < 1e-6
