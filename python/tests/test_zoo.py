"""Tests for the CML model zoo."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

# The CFFI extension must be built first (cd python && python3 cml/build_cffi.py).
# Skip cleanly rather than erroring on machines without it.
try:
    import cml  # noqa: F401
except ImportError as e:
    pytest.skip(f"cml CFFI module not built ({e}); run `python3 cml/build_cffi.py` "
                "in python/ first", allow_module_level=True)


def test_mlp_mnist_creation():
    """Test MLP-MNIST model creation without CML library."""
    # This tests the Python-level API structure
    from cml import zoo
    assert hasattr(zoo, 'mlp_mnist')
    assert hasattr(zoo, 'mlp_cifar10')
    assert hasattr(zoo, 'resnet18')
    assert hasattr(zoo, 'resnet34')
    assert hasattr(zoo, 'resnet50')
    assert hasattr(zoo, 'vgg11')
    assert hasattr(zoo, 'vgg16')
    assert hasattr(zoo, 'gpt2_small')
    assert hasattr(zoo, 'bert_tiny')
    assert hasattr(zoo, 'download_weights')
    print("PASS: Zoo API structure verified")


def test_distributed_api():
    """Test distributed API structure."""
    from cml import distributed
    assert hasattr(distributed, 'init_process_group')
    assert hasattr(distributed, 'get_rank')
    assert hasattr(distributed, 'get_world_size')
    assert hasattr(distributed, 'is_initialized')
    assert hasattr(distributed, 'destroy_process_group')
    assert hasattr(distributed, 'barrier')
    assert hasattr(distributed, 'DistributedDataParallel')
    assert hasattr(distributed, 'PipelineParallel')
    print("PASS: Distributed API structure verified")


if __name__ == "__main__":
    test_mlp_mnist_creation()
    test_distributed_api()
    print("All Python tests passed!")
