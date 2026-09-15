"""ONNX export for C-ML graphs.

Exports the IR graph behind one or more result tensors to a standard ONNX
model file (opset 11). Weights (eager leaf tensors) become initializers;
placeholder tensors passed via ``inputs`` become named graph inputs
(``input_0``, ``input_1``, ...).

Example:
    x = cml.Tensor(np.random.randn(2, 3).astype(np.float32))
    w = cml.nn.Linear(3, 4)
    y = w(x)
    cml.onnx.export([y], "model.onnx", inputs=[x])

Ops with no ONNX equivalent raise ``OnnxExportError`` naming the op rather
than writing a wrong model.
"""

from __future__ import annotations

from pathlib import Path

from cml._cml_lib import ffi, lib


class OnnxExportError(RuntimeError):
    """Raised when a graph contains an operation ONNX cannot represent."""


def export(outputs: list, path: str | Path, inputs: list | None = None) -> None:
    """Export the graph computing ``outputs`` to ``path`` in ONNX format.

    Args:
        outputs: result tensors; each must belong to the same IR context.
        path: destination ``.onnx`` file path.
        inputs: placeholder tensors that should appear as named model inputs.
                Anything else with data (weights, constants) is exported as
                an initializer.
    """
    if not outputs:
        raise ValueError("outputs must contain at least one tensor")
    inputs = list(inputs) if inputs else []

    out_arr = ffi.new("Tensor*[]", [t._tensor for t in outputs])
    in_arr = (
        ffi.new("Tensor*[]", [t._tensor for t in inputs]) if inputs else ffi.NULL
    )
    ir = outputs[0]._tensor.ir_context
    if ir == ffi.NULL or ir is None:
        raise OnnxExportError(
            "output tensor has no IR context (is it an eager leaf?)"
        )

    rc = lib.cml_onnx_export_graph(
        ir,
        in_arr,
        len(inputs),
        out_arr,
        len(outputs),
        str(path).encode("utf-8"),
    )
    if rc != 0:
        raise OnnxExportError(
            f"ONNX export failed for '{path}' -- see log for the "
            "unsupported operation or serialization error"
        )


def supported(op_type: str) -> bool:
    """Whether the C-ML ONNX runtime can execute the given operator."""
    return bool(lib.cml_onnx_op_supported(op_type.encode("utf-8")))
