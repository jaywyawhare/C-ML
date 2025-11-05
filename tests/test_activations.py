import numpy as np
import torch


def test_activations(opout):
    x = torch.tensor([-1.0, -0.5, 0.25, 2.0], dtype=torch.float32)
    np.testing.assert_allclose(
        opout["RELU"], torch.nn.functional.relu(x).numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        opout["SIGMOID"], torch.sigmoid(x).numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        opout["TANH"], torch.tanh(x).numpy(), rtol=1e-6, atol=1e-6
    )
