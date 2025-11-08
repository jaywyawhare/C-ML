import torch


def test_losses(opout):
    x = torch.tensor([-1.0, -0.5, 0.25, 2.0], dtype=torch.float32)
    tgt = torch.tensor([0.0, 0.0, 0.5, 1.5], dtype=torch.float32)
    relu = torch.nn.functional.relu(x)
    mse = torch.mean((relu - tgt) ** 2).item()
    mae = torch.mean(torch.abs(relu - tgt)).item()
    assert abs(opout["MSE"] - mse) < 1e-6
    assert abs(opout["MAE"] - mae) < 1e-6

    probs = torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    bce = torch.nn.functional.binary_cross_entropy(probs, labels).item()
    assert abs(opout["BCE"] - bce) < 5e-6
