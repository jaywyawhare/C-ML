"""Runtime module loading and execution (.cpte / AOT)."""

from __future__ import annotations
from typing import Optional
from cml._cml_lib import ffi, lib
from cml.core import Tensor, Module
from cml.torch.memory import MemoryManager

_DEFAULT_FORWARD_METHOD = ffi.new("char[]", b"forward")


class RuntimeModule:
    """ExecuTorch-style loaded program wrapper for eager, PTE, or AOT backends."""

    def __init__(self, ptr):
        """Wrap an existing native ``TorchRuntimeModule`` pointer.

        Args:
            ptr: CFFI pointer returned by the torch_c runtime loaders.
        """
        self._rt = ptr
        self._memory: Optional[MemoryManager] = None

    @classmethod
    def from_module(cls, module: Module) -> "RuntimeModule":
        """Wrap an in-memory eager ``Module`` without taking ownership.

        Args:
            module: Trained or initialized C-ML module.

        Returns:
            A runtime handle that forwards through ``module_forward``.

        Raises:
            RuntimeError: If the native wrapper cannot be created.
        """
        rt = lib.torch_runtime_from_module(module._module)
        if rt == ffi.NULL:
            raise RuntimeError("Failed to wrap module")
        return cls(rt)

    @classmethod
    def load_pte(cls, path: str) -> "RuntimeModule":
        """Load a portable ``.cpte`` program from disk.

        Args:
            path: File path to the serialized PTE program.

        Returns:
            A runtime ready for ``forward`` calls.

        Raises:
            RuntimeError: If loading fails.
        """
        rt = lib.torch_runtime_load_pte(path.encode())
        if rt == ffi.NULL:
            raise RuntimeError(f"Failed to load PTE: {path}")
        return cls(rt)

    @classmethod
    def load_aot(cls, path: str) -> "RuntimeModule":
        """Load an ahead-of-time compiled shared library.

        Args:
            path: Path to the AOT model artifact.

        Returns:
            A runtime ready for ``forward`` calls.

        Raises:
            RuntimeError: If loading fails.
        """
        rt = lib.torch_runtime_load_aot(path.encode())
        if rt == ffi.NULL:
            raise RuntimeError(f"Failed to load AOT model: {path}")
        return cls(rt)

    def set_memory(self, memory: MemoryManager) -> None:
        """Attach a user-provided arena for PTE execution.

        Args:
            memory: Arena manager; must remain open for the runtime lifetime.
        """
        lib.torch_runtime_set_memory(self._rt, memory._mgr)
        self._memory = memory

    def forward(self, x: Tensor) -> Tensor:
        """Run the loaded program on an input tensor.

        Args:
            x: Input activation tensor.

        Returns:
            Output tensor produced by the runtime.

        Raises:
            RuntimeError: If native forward fails.
        """
        out = lib.torch_runtime_forward(self._rt, x._tensor)
        if out == ffi.NULL:
            raise RuntimeError("Runtime forward failed")
        return Tensor(out)

    def __del__(self):
        """Free the native runtime handle."""
        if hasattr(self, "_rt") and self._rt != ffi.NULL:
            lib.torch_runtime_free(self._rt)
            self._rt = ffi.NULL


def export_pte(module: Module, sample_input: Tensor, path: str,
               include_weights: bool = True) -> None:
    """Export an eager module to a portable ``.cpte`` program.

    Args:
        module: Module to trace and serialize.
        sample_input: Representative input used to capture the forward graph.
        path: Output file path for the ``.cpte`` artifact.
        include_weights: Whether parameter tensors are embedded in the file.

    Raises:
        RuntimeError: If export fails.
    """
    opts = ffi.new("TorchPTEExportOptions*")
    opts.method_name = _DEFAULT_FORWARD_METHOD
    opts.backend = 0
    opts.include_weights = include_weights
    opts.compute_memory_plan = True
    opts.aot_output_path = ffi.NULL
    rc = lib.torch_runtime_export_pte(
        module._module, sample_input._tensor, path.encode(), opts)
    if rc != 0:
        raise RuntimeError(f"PTE export failed: {path}")


def load_pte(path: str) -> RuntimeModule:
    """Load a ``.cpte`` program from ``path``.

    Args:
        path: File path to the serialized program.

    Returns:
        A ``RuntimeModule`` ready for inference.
    """
    return RuntimeModule.load_pte(path)


def load_aot(path: str) -> RuntimeModule:
    """Load an AOT-compiled model from ``path``.

    Args:
        path: Path to the AOT shared library or artifact.

    Returns:
        A ``RuntimeModule`` ready for inference.
    """
    return RuntimeModule.load_aot(path)
