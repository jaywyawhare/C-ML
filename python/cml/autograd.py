"""
Automatic differentiation support.
"""

from cml._cml_lib import ffi, lib
from cml.core import Tensor


def backward(loss, grad=None, clear_grads=False, retain_graph=False):
    """Compute gradients via backpropagation.

    Args:
        loss: Loss tensor (usually scalar)
        grad: Gradient with respect to loss (None for scalar with grad=1)
        clear_grads: Whether to clear existing gradients before backward pass
        retain_graph: Keep computation graph for multiple backward passes

    Example:
        >>> output = model(x)
        >>> loss = mse_loss(output, target)
        >>> backward(loss)  # Compute gradients
        >>> optimizer.step()  # Update parameters
    """
    grad_ptr = grad._tensor if grad is not None else ffi.NULL
    lib.cml_backward(loss._tensor, grad_ptr, clear_grads, retain_graph)


def get_grad(tensor):
    """Get computed gradients for a tensor.

    Args:
        tensor: Tensor to get gradients for

    Returns:
        Gradient tensor or None if no gradients

    Example:
        >>> x = randn([10, 10])
        >>> y = x * 2
        >>> loss = y.sum()
        >>> backward(loss)
        >>> grad_x = get_grad(x)
    """
    grad_ptr = lib.cml_get_grad(tensor._tensor)
    if grad_ptr == ffi.NULL:
        return None
    return Tensor(grad_ptr)


def enable_grad(tensor):
    """Enable gradient computation for a tensor.

    Args:
        tensor: Tensor to enable gradients for

    Example:
        >>> x = randn([10, 10])
        >>> enable_grad(x)
        >>> # Now gradients will be computed for x
    """
    lib.cml_enable_grad(tensor._tensor)


def disable_grad(tensor):
    """Disable gradient computation for a tensor.

    Args:
        tensor: Tensor to disable gradients for

    Example:
        >>> x = randn([10, 10])
        >>> disable_grad(x)
        >>> # Now gradients won't be computed for x
    """
    lib.cml_disable_grad(tensor._tensor)


class no_grad:
    """Context manager to disable gradient computation.

    Example:
        >>> with no_grad():
        ...     output = model(x)  # Gradients not computed
    """

    def __enter__(self):
        """Enter no_grad context."""
        self._grads_enabled = True  # Track state if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit no_grad context."""
        return False
