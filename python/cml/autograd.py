"""Automatic differentiation support."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor, _TensorView

# Detaching the loss after backward (and resetting the graph in optimizer.step)
# keeps a hand-written training loop from piling every step's autograd graph onto
# the reused parameters. Feature-detected so an un-rebuilt extension still works.
_HAS_REALIZE = hasattr(lib, "tensor_realize")


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
    # Detach (materialize) the loss so its value stays readable and it survives
    # the graph reset that optimizer.step() performs. Skipped when the caller
    # wants to keep the graph for another backward.
    if _HAS_REALIZE and not retain_graph and not create_graph \
            and loss is not None and loss._tensor != ffi.NULL:
        lib.tensor_realize(loss._tensor)


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
    """Zero the accumulated gradient of a tensor in place."""
    lib.cml_zero_grad(tensor._tensor)


def requires_grad(tensor):
    """Return whether the tensor currently tracks gradients."""
    return lib.cml_requires_grad(tensor._tensor)


def is_leaf(tensor):
    """Return whether the tensor is a graph leaf (has no autograd history)."""
    return lib.cml_is_leaf(tensor._tensor)


# ── Custom differentiable ops (torch.autograd.Function equivalent) ──────────
#
# Approach: the C layer (cml_autograd_function_apply) owns the graph integration;
# Python forward/backward run through CFFI callback thunks bridged to it. The
# thunks and any tensors they must keep alive are held in `_FUNCTION_KEEPALIVE`
# until the process exits or Function.reset() is called -- a dropped callback
# would leave the C node holding a dangling function pointer for the backward.
#
# ctx.save_for_backward / ctx.saved_tensors round-trip through the C context;
# arbitrary Python scratch can also be stashed as plain ctx attributes.
#
# Double-backward (create_graph) is NOT supported across a custom op: the Python
# backward returns concrete gradient tensors, not a differentiable graph.

_FUNCTION_KEEPALIVE = []


class _Ctx:
    """The context handed to a Function's forward/backward (torch-style)."""

    def __init__(self, c_ctx, holder):
        self._c_ctx = c_ctx
        self._holder = holder

    def save_for_backward(self, *tensors):
        """Stash tensors for the backward pass."""
        self._holder["saved_py"] = list(tensors)
        if not tensors:
            lib.cml_autograd_ctx_save(self._c_ctx, ffi.NULL, 0)
            return
        arr = ffi.new("Tensor*[]", [t._tensor for t in tensors])
        lib.cml_autograd_ctx_save(self._c_ctx, arr, len(tensors))

    @property
    def saved_tensors(self):
        """Return the tensors stashed by save_for_backward."""
        saved = self._holder.get("saved_py")
        if saved is not None:
            return tuple(saved)
        out_n = ffi.new("int*")
        ptr = lib.cml_autograd_ctx_saved(self._c_ctx, out_n)
        n = out_n[0]
        if ptr == ffi.NULL or n == 0:
            return ()
        return tuple(Tensor.borrow(ptr[i]) for i in range(n))


class Function:
    """Base class for a custom differentiable op (cf. torch.autograd.Function).

    Subclass and define static ``forward(ctx, *inputs)`` and
    ``backward(ctx, grad_output)``. Call ``MyFn.apply(*inputs)`` to run it inside
    the autograd graph. ``forward`` returns the output tensor; ``backward``
    returns one gradient tensor per input (a single tensor or a tuple), using
    ``None`` for inputs that need no gradient.
    """

    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError("Function subclasses must define forward()")

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Function subclasses must define backward()")

    @classmethod
    def apply(cls, *inputs):
        holder = {"saved_py": None, "grad_keep": [], "out_py": None, "cbs": None}

        @ffi.callback("Tensor*(CMLAutogradContext*, Tensor**, int, void*)")
        def _fwd(c_ctx, c_inputs, n, _user_data):
            ctx = _Ctx(c_ctx, holder)
            py_inputs = [Tensor.borrow(c_inputs[i]) for i in range(n)]
            out = cls.forward(ctx, *py_inputs)
            if out is None:
                return ffi.NULL
            # Hold a reference so the output's pin survives until apply() below
            # re-wraps and returns it (the C node keeps only a raw pointer).
            holder["out_py"] = out
            return out._tensor

        @ffi.callback("void(CMLAutogradContext*, Tensor*, Tensor**, int, void*)")
        def _bwd(c_ctx, grad_output, grad_inputs, n, _user_data):
            ctx = _Ctx(c_ctx, holder)
            grads = cls.backward(ctx, Tensor.borrow(grad_output))
            if not isinstance(grads, (tuple, list)):
                grads = (grads,)
            for i in range(n):
                g = grads[i] if i < len(grads) else None
                if g is None:
                    grad_inputs[i] = ffi.NULL
                else:
                    holder["grad_keep"].append(g)  # accumulate_grad clones it
                    grad_inputs[i] = g._tensor

        holder["cbs"] = (_fwd, _bwd)
        _FUNCTION_KEEPALIVE.append(holder)

        c_inputs = ffi.new("Tensor*[]", [t._tensor for t in inputs]) if inputs else ffi.NULL
        out_ptr = lib.cml_autograd_function_apply(_fwd, _bwd, c_inputs, len(inputs), ffi.NULL)
        if out_ptr == ffi.NULL:
            return None
        produced = holder.get("out_py")
        if produced is not None and produced._tensor == out_ptr:
            return produced
        return Tensor(out_ptr)

    @staticmethod
    def reset():
        """Drop all registered custom nodes and the callbacks kept alive for them."""
        lib.cml_autograd_custom_reset()
        _FUNCTION_KEEPALIVE.clear()
