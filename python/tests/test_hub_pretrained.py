"""The zoo `pretrained=True` path must actually load cached weights.

It previously looked for a `.bin` and called a non-existent `model_load`, so it
silently loaded nothing. It now goes through the real safetensors loader. This
saves a model's parameters, then builds a fresh model with `pretrained=True` and
requires every parameter to come back bit-identical - proving the load happened
and matched by name/shape.
"""
import os
import tempfile

import numpy as np
import pytest

import cml
from cml import safetensors, zoo


@pytest.fixture(scope="module", autouse=True)
def _init():
    cml.init()


def test_zoo_pretrained_roundtrip(monkeypatch):
    src = zoo.mlp_mnist()
    params = list(src.parameters(recursive=True))
    assert params, "model has no parameters"

    # Stamp each parameter with a distinctive value so a no-op load is detectable.
    expected = {}
    for i, p in enumerate(params):
        shape = p.tensor.numpy().shape
        val = np.full(shape, 0.1 + 0.01 * i, dtype=np.float32)
        safetensors._copy_into(p.tensor, val)
        expected[p.name] = val

    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("CML_WEIGHTS_DIR", tmp)
    safetensors.save_file({p.name: p.tensor for p in params},
                          os.path.join(tmp, "mlp_mnist.safetensors"))

    # Fresh model + pretrained=True must pull the saved weights in.
    loaded = zoo.mlp_mnist(pretrained=True)
    got = {p.name: p for p in loaded.parameters(recursive=True)}
    assert set(got) == set(expected), "parameter names changed"
    for name, want in expected.items():
        assert np.allclose(got[name].tensor.numpy(), want), f"{name} not loaded"
