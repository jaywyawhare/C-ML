"""PyTorch-like namespace for C-ML (ExecuTorch-inspired runtime façade)."""

from cml.torch.runtime import RuntimeModule, export_pte, load_pte, load_aot
from cml.torch.memory import MemoryManager
from cml.torch.tensor import zeros, ones, randn, options
from cml.torch.eager import (
    set_eager_mode,
    is_eager_mode,
    inference_mode,
    set_num_threads,
    get_num_threads,
    realize,
    linear,
    linear_relu,
)

__all__ = [
    "RuntimeModule",
    "export_pte",
    "load_pte",
    "load_aot",
    "MemoryManager",
    "zeros",
    "ones",
    "randn",
    "options",
    "set_eager_mode",
    "is_eager_mode",
    "inference_mode",
    "set_num_threads",
    "get_num_threads",
    "realize",
    "linear",
    "linear_relu",
]
