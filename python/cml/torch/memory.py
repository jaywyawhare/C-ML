"""User-provided memory arenas for edge deployment."""

from __future__ import annotations
from cml._cml_lib import ffi, lib


class MemoryManager:
    """Wraps a C-ML ``TorchMemoryManager`` bump-allocator arena.

    Arenas provide deterministic, caller-controlled memory for edge deployment.
    Not thread-safe: do not share one instance across threads while another
    thread may call ``close()`` or rely on garbage-collection teardown.
    Prefer explicit ``close()`` or a ``with`` block in multi-threaded hosts.
    """

    def __init__(self, size: int):
        """Allocate an owned arena of ``size`` bytes.

        Args:
            size: Number of bytes reserved for bump allocation.

        Raises:
            MemoryError: If the native arena cannot be created.
        """
        self._buffer = None
        self._buffer_cdata = None
        self._mgr = lib.torch_memory_create(size)
        if self._mgr == ffi.NULL:
            raise MemoryError(f"Failed to allocate {size}-byte arena")

    @classmethod
    def from_buffer(cls, buffer, size: int) -> "MemoryManager":
        """Wrap caller-owned memory without taking ownership of the buffer.

        Args:
            buffer: Writable buffer object (e.g. ``bytearray``, ``memoryview``).
            size: Number of valid bytes in ``buffer``.

        Returns:
            A ``MemoryManager`` that allocates from the supplied storage.

        Raises:
            MemoryError: If the native wrapper cannot be created.
        """
        mv = memoryview(buffer)
        if mv.readonly:
            raise ValueError("buffer must be writable")
        if size <= 0 or mv.nbytes < size:
            raise ValueError("size exceeds backing buffer length")
        mgr = cls.__new__(cls)
        mgr._buffer = mv
        mgr._buffer_cdata = ffi.from_buffer(mv)
        mgr._mgr = lib.torch_memory_from_buffer(mgr._buffer_cdata, size)
        if mgr._mgr == ffi.NULL:
            raise MemoryError("Failed to wrap external buffer")
        return mgr

    def _require_open(self) -> None:
        """Raise if this manager has already been closed."""
        if self._mgr == ffi.NULL:
            raise RuntimeError("MemoryManager is closed")

    @property
    def used(self) -> int:
        """Current number of bytes allocated from the arena."""
        self._require_open()
        return int(lib.torch_memory_used(self._mgr))

    @property
    def peak(self) -> int:
        """Peak bytes allocated from the arena since creation or last reset."""
        self._require_open()
        return int(lib.torch_memory_peak(self._mgr))

    def close(self) -> None:
        """Release the native arena and invalidate this manager."""
        if getattr(self, "_mgr", ffi.NULL) != ffi.NULL:
            lib.torch_memory_free(self._mgr)
            self._mgr = ffi.NULL
        self._buffer_cdata = None
        self._buffer = None

    def __enter__(self) -> "MemoryManager":
        """Enter a context block without changing arena state."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the arena when leaving a ``with`` block."""
        self.close()

    def __del__(self):
        """Close the arena during garbage collection."""
        self.close()
