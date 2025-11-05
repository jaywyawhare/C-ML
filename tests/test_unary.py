import numpy as np
import torch


def test_unary_math(opout):
    x1 = torch.tensor([0.1, 1.0, 2.0, 4.0], dtype=torch.float32)
    np.testing.assert_allclose(
        opout["EXP"], torch.exp(x1).numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        opout["LOG"], torch.log(x1).numpy(), rtol=1e-6, atol=5e-6
    )
    np.testing.assert_allclose(
        opout["SQRT"], torch.sqrt(x1).numpy(), rtol=1e-6, atol=5e-6
    )

    x2 = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float32)
    np.testing.assert_allclose(
        opout["SIN"], torch.sin(x2).numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        opout["COS"], torch.cos(x2).numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        opout["TAN"], torch.tan(x2).numpy(), rtol=1e-5, atol=1e-5
    )
