import numpy as np
import torch


def test_elementwise_ops(opout):
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    b = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32)

    np.testing.assert_allclose(opout["ADD"], (a + b).numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(opout["SUB"], (b - a).numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(opout["MUL"], (a * b).numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(opout["DIV"], (b / a).numpy(), rtol=1e-6, atol=5e-6)

    exps = torch.tensor([1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(opout["POW"], (a**exps).numpy(), rtol=1e-6, atol=1e-6)
