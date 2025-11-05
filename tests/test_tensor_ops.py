import numpy as np
import torch


def test_tensor_ops(opout):
    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    B = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    mm = (A @ B).flatten().numpy()
    At = A.t().flatten().numpy()
    np.testing.assert_allclose(opout["MATMUL"], mm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(opout["TRANSPOSE"], At, rtol=1e-6, atol=1e-6)
