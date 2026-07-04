"""Zero-IR eager execution + GEMM thread tuning + fused linear."""

from __future__ import annotations
from contextlib import contextmanager
from cml._cml_lib import ffi, lib
from cml.core import Tensor


def set_eager_mode(enabled: bool) -> None:
    """Enable/disable zero-IR eager execution of hot ops (add/mul/matmul/relu...)."""
    lib.torch_set_eager_mode(bool(enabled))


def is_eager_mode() -> bool:
    """Return whether zero-IR eager execution is currently enabled."""
    return bool(lib.torch_is_eager_mode())


def set_num_threads(n: int) -> None:
    """Set the global BLAS (MKL/OpenBLAS/BLIS/ILP64) thread count for GEMM."""
    lib.torch_set_num_threads(int(n))


def get_num_threads() -> int:
    """Return the process-global BLAS thread count used for GEMM."""
    return int(lib.torch_get_num_threads())


def realize(t: Tensor) -> Tensor:
    """Materialize and detach a tensor from the IR graph (survives resets)."""
    rc = lib.torch_realize(t._tensor)
    if rc != 0:
        raise RuntimeError("torch_realize failed")
    return t


@contextmanager
def inference_mode():
    """Context manager: eager + no_grad for the duration, restored on exit."""
    lib.torch_inference_mode(True)
    try:
        yield
    finally:
        lib.torch_inference_mode(False)


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Fused out = input @ weight^T + bias (single BLAS GEMM + fused bias)."""
    b = bias._tensor if bias is not None else ffi.NULL
    out = lib.torch_linear(input._tensor, weight._tensor, b)
    if out == ffi.NULL:
        raise RuntimeError("torch_linear failed")
    return Tensor(out)


def linear_relu(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Fused out = relu(input @ weight^T + bias) (single GEMM + fused bias+relu)."""
    b = bias._tensor if bias is not None else ffi.NULL
    out = lib.torch_linear_relu(input._tensor, weight._tensor, b)
    if out == ffi.NULL:
        raise RuntimeError("torch_linear_relu failed")
    return Tensor(out)
