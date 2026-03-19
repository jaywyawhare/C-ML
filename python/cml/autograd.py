"""Automatic differentiation support."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor, _TensorView


def backward(loss, grad=None, retain_graph=False, create_graph=False):
    """Run backward pass on a loss tensor.

    Args:
        loss: The loss tensor to differentiate.
        grad: Optional gradient tensor (defaults to ones).
        retain_graph: If True, keep the computation graph for further backward passes.
        create_graph: If True, create a graph of the derivative for higher-order gradients.
    """
    grad_ptr = grad._tensor if grad is not None else ffi.NULL
    lib.cml_backward(loss._tensor, grad_ptr, retain_graph, create_graph)


def get_grad(tensor):
    """Get the gradient of a tensor by accessing its grad field directly."""
    if tensor._tensor == ffi.NULL:
        return None
    grad_ptr = tensor._tensor.grad
    if grad_ptr == ffi.NULL:
        return None
    return _TensorView(grad_ptr)


def enable_grad(tensor=None):
    """Enable requires_grad on a tensor, or the global gradient flag if none given."""
    if tensor is not None:
        lib.cml_set_requires_grad(tensor._tensor, True)
    else:
        lib.cml_enable_grad()


def disable_grad(tensor=None):
    """Disable requires_grad on a tensor, or the global gradient flag if none given."""
    if tensor is not None:
        lib.cml_set_requires_grad(tensor._tensor, False)
    else:
        lib.cml_no_grad()


def zero_grad(tensor):
    lib.cml_zero_grad(tensor._tensor)


def requires_grad(tensor):
    return lib.cml_requires_grad(tensor._tensor)


def is_leaf(tensor):
    return lib.cml_is_leaf(tensor._tensor)
