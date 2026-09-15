"""Tests for ONNX export (python/cml/onnx.py).

The C suite (tests/test_onnx_export.c) validates numeric round-trips through
the independent importer; here we pin the Python API surface and the
fail-loud contract.
"""

import os

import numpy as np
import pytest

import cml
import cml.nn as cnn
from cml.onnx import OnnxExportError, export


def _tmp(tmp_path, name="model.onnx"):
    return str(tmp_path / name)


@pytest.fixture(autouse=True)
def _fresh_ir_graph():
    """Export serializes every node in the process-global IR graph; drop
    nodes left by earlier tests so each export sees only its own subgraph."""
    cml.reset_graph()
    yield
    cml.reset_graph()


def test_export_linear_writes_file(tmp_path):
    np.random.seed(0)
    x = cml.Tensor(np.random.randn(2, 3).astype(np.float32))
    lin = cnn.Linear(3, 4)
    y = lin(x)
    path = _tmp(tmp_path)
    export([y], path, inputs=[x])
    size = os.path.getsize(path)
    assert size > 50


def test_export_requires_ir_context():
    x = cml.Tensor(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(OnnxExportError):
        export([x], "/tmp/cml_should_not_exist.onnx")


def test_export_empty_outputs_raises():
    with pytest.raises(ValueError):
        export([], "/tmp/cml_should_not_exist.onnx")
