import numpy as np
import pytest
import torch


def test_linear_forward(opout):
    inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    W = torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.4]], dtype=torch.float32)
    b = torch.tensor([0.01, -0.03], dtype=torch.float32)
    linear = (inp @ W.t()) + b
    np.testing.assert_allclose(
        opout["LINEAR"], linear.flatten().numpy(), rtol=1e-5, atol=1e-5
    )


def test_conv2d_forward(opout):
    # 1x1 conv identity-like on [1,1,2,2] -> equal to input
    inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(opout["CONV2D"], inp, rtol=1e-6, atol=1e-6)


def test_batchnorm2d_forward(opout):
    # BN2D over N=1,C=1,H=2,W=2 with values [1,2,3,4]
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    mean = torch.mean(x)
    var = torch.var(x, unbiased=False)
    eps = 1e-5
    bn = ((x - mean) / torch.sqrt(var + eps)).numpy()
    np.testing.assert_allclose(opout["BN2D"], bn, rtol=1e-5, atol=1e-5)


def test_pooling_forward(opout):
    # MaxPool2d/AvgPool2d kernel=2 on [1,1,2,2] values [1,2;3,4]
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    maxp = (
        torch.nn.functional.max_pool2d(x.unsqueeze(0).unsqueeze(0), 2, 2, 0)
        .flatten()
        .numpy()
    )
    avgp = (
        torch.nn.functional.avg_pool2d(
            x.unsqueeze(0).unsqueeze(0), 2, 2, 0, count_include_pad=True
        )
        .flatten()
        .numpy()
    )
    np.testing.assert_allclose(opout["MAXPOOL"], maxp, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(opout["AVGPOOL"], avgp, rtol=1e-6, atol=1e-6)
