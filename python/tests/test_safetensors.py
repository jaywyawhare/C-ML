"""Tests for safetensors save/load and HF-style weight loading."""

import numpy as np
import pytest

import cml
import cml.nn as cnn
from cml.safetensors import load_file, load_pretrained, save_file


def test_roundtrip(tmp_path):
    d = {
        "w": np.random.randn(4, 4).astype(np.float32),
        "b": np.arange(4, dtype=np.float32),
        "i": np.arange(6, dtype=np.int32).reshape(2, 3),
    }
    path = str(tmp_path / "model.safetensors")
    save_file(d, path)
    out = load_file(path)
    for k in d:
        assert out[k].numpy().shape == d[k].shape
        assert np.allclose(out[k].numpy(), d[k])


def test_header_alignment(tmp_path):
    """Spec: the payload must start at an 8-byte-aligned offset."""
    import json
    import struct

    path = str(tmp_path / "a.safetensors")
    save_file({"x": np.ones(3, dtype=np.float32)}, path)
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        assert hlen % 8 == 0
        json.loads(f.read(hlen))  # header must be valid JSON


def test_load_pretrained_mapping():
    l1 = cnn.Linear(4, 2)
    seq = cnn.Sequential(l1)
    state = {
        "layers.0.weight": np.random.randn(2, 4).astype(np.float32),
        "layers.0.bias": np.random.randn(2).astype(np.float32),
    }
    n = load_pretrained(seq, state, mapping={"layers.0.": "0.Linear."})
    assert n == 2
    w = l1.parameters()[0].tensor.numpy()
    assert np.allclose(w, state["layers.0.weight"])
    b = l1.parameters()[1].tensor.numpy()
    assert np.allclose(b, state["layers.0.bias"])


def test_load_pretrained_strict_missing():
    l1 = cnn.Linear(4, 2)
    seq = cnn.Sequential(l1)
    # entry with no matching parameter -> strict must raise
    state = {"encoder.weight": np.random.randn(2, 4).astype(np.float32)}
    with pytest.raises(KeyError):
        load_pretrained(seq, state, mapping={"layers.0.": "0.Linear."},
                        strict=True)


def test_load_pretrained_shape_mismatch_strict():
    l1 = cnn.Linear(4, 2)
    seq = cnn.Sequential(l1)
    state = {"layers.0.weight": np.random.randn(8, 4).astype(np.float32)}
    with pytest.raises(ValueError):
        load_pretrained(seq, state, mapping={"layers.0.": "0.Linear."},
                        strict=True)
