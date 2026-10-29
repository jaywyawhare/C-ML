"""CSR / CSC sparse tensor support (torch.sparse-style helpers).

Thin wrappers over the C sparse layer (``tensor/sparse_tensor.h``). A
``SparseCSRTensor`` owns a C ``SparseCSRData`` handle; conversions and the
differentiable SpMM (``matmul``) return plain :class:`~cml.core.Tensor`
objects.
"""

from __future__ import annotations

from typing import Optional

from cml._cml_lib import ffi, lib
from cml.core import Tensor, _make_config


def _wrap_tensor(ptr):
    if ptr == ffi.NULL:
        raise RuntimeError("sparse op returned NULL")
    return Tensor(ptr)


class SparseCSRTensor:
    """A CSR sparse matrix backed by a C ``SparseCSRData`` handle."""

    def __init__(self, handle):
        if handle == ffi.NULL:
            raise RuntimeError("failed to construct SparseCSRTensor")
        self._csr = handle

    def __del__(self):
        if getattr(self, "_csr", None) not in (None, ffi.NULL):
            lib.sparse_csr_free(self._csr)
            self._csr = ffi.NULL

    @property
    def shape(self):
        return (self._csr.rows, self._csr.cols)

    @property
    def nnz(self) -> int:
        return lib.sparse_csr_nnz(self._csr)

    @property
    def density(self) -> float:
        return lib.sparse_csr_density(self._csr)

    @property
    def is_sparse(self) -> bool:
        return bool(lib.sparse_csr_is_sparse(self._csr))

    def to_dense(self, dtype=None, device=None) -> Tensor:
        cfg = _make_config(dtype, device)
        return _wrap_tensor(lib.sparse_csr_to_dense(self._csr, cfg))

    def matmul(self, dense: Tensor) -> Tensor:
        """SpMM: ``self @ dense`` -> dense. Differentiable wrt ``dense`` and
        this matrix's values when either requires grad."""
        return _wrap_tensor(lib.sparse_csr_matmul(self._csr, dense._tensor))

    def mv(self, vec: Tensor) -> Tensor:
        return _wrap_tensor(lib.sparse_mv(self._csr, vec._tensor))

    def add(self, other: "SparseCSRTensor") -> "SparseCSRTensor":
        return SparseCSRTensor(lib.sparse_add(self._csr, other._csr))

    def mul_scalar(self, scalar: float) -> "SparseCSRTensor":
        return SparseCSRTensor(lib.sparse_mul_scalar(self._csr, float(scalar)))

    def transpose(self) -> "SparseCSCTensor":
        return SparseCSCTensor(lib.sparse_transpose(self._csr))

    def sum(self, dim: Optional[int] = None, dtype=None, device=None) -> Tensor:
        cfg = _make_config(dtype, device)
        return _wrap_tensor(lib.sparse_sum(self._csr, -1 if dim is None else dim, cfg))


class SparseCSCTensor:
    """A CSC sparse matrix backed by a C ``SparseCSCData`` handle."""

    def __init__(self, handle):
        if handle == ffi.NULL:
            raise RuntimeError("failed to construct SparseCSCTensor")
        self._csc = handle

    def __del__(self):
        if getattr(self, "_csc", None) not in (None, ffi.NULL):
            lib.sparse_csc_free(self._csc)
            self._csc = ffi.NULL

    @property
    def shape(self):
        return (self._csc.rows, self._csc.cols)

    @property
    def nnz(self) -> int:
        return self._csc.nnz


def to_sparse_csr(dense: Tensor) -> SparseCSRTensor:
    """Convert a dense 2-D tensor to CSR (torch's ``Tensor.to_sparse_csr``)."""
    return SparseCSRTensor(lib.dense_to_sparse_csr(dense._tensor))


def sparse_csr_tensor(crow_indices: Tensor, col_indices: Tensor, values: Tensor,
                      size) -> SparseCSRTensor:
    """Build a CSR tensor from row pointers, column indices and values.

    Mirrors ``torch.sparse_csr_tensor``; ``size`` is ``(rows, cols)``.
    """
    rows, cols = int(size[0]), int(size[1])
    return SparseCSRTensor(
        lib.sparse_csr_create(crow_indices._tensor, col_indices._tensor,
                              values._tensor, rows, cols))
