"""Automatic differentiation support."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor


def backward(loss, grad=None, clear_grads=False, retain_graph=False):
    grad_ptr = grad._tensor if grad is not None else ffi.NULL
    lib.cml_backward(loss._tensor, grad_ptr, clear_grads, retain_graph)


def get_grad(tensor):
    grad_ptr = lib.cml_get_grad(tensor._tensor)
    if grad_ptr == ffi.NULL:
        return None
    return Tensor(grad_ptr)


def enable_grad(tensor):
    lib.cml_enable_grad(tensor._tensor)


def disable_grad(tensor):
    lib.cml_disable_grad(tensor._tensor)


class no_grad:
    def __enter__(self):
        self._grads_enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
