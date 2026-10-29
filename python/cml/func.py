"""torch.func-style functional transforms for C-ML.

These wrap the C transform family in ``autograd/functional_transforms.h`` and
``autograd/forward_mode.h`` directly: a user's Python function is bridged to the
C ``CMLTransformFn`` / ``CMLForwardFn`` callback via ``ffi.callback``, so the
autodiff itself always runs in C (nothing is reimplemented in Python).

Engine support mirrors the C layer. The backward engine is chosen once per
process by ``GRAD_MODE`` (graph is the default):

* ``grad``, ``vjp``, ``jacrev``, ``vmap``, ``functional_call`` -- reverse mode or
  pure slicing; work on both the graph and eager engines.
* ``jvp``, ``jacfwd``, ``hessian`` -- forward-over-reverse / reverse-over-reverse;
  require the graph engine and raise ``RuntimeError`` under ``GRAD_MODE=eager``.

Callback lifetime: input tensors are wrapped as non-owning ``Tensor.borrow``
views; outputs produced inside a callback are held alive for the duration of the
enclosing C call so the garbage collector cannot free them mid-transform.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple, Union

from cml._cml_lib import ffi, lib
from cml.core import Tensor

_TRANSFORM_SIG = "Tensor*(Tensor**, int, void*)"
_FORWARD_SIG = "Tensor*(Tensor*, void*)"


def _uses_graph() -> bool:
    return bool(lib.cml_autodiff_use_graph())


def _require_graph(name: str) -> None:
    if not _uses_graph():
        raise RuntimeError(
            f"cml.func.{name} requires the graph autodiff engine; it is unavailable "
            "under GRAD_MODE=eager (forward/double-mode needs differentiable grads)"
        )


def _wrap(ptr):
    return Tensor(ptr) if ptr != ffi.NULL else None


def _transform_callback(func: Callable[..., Tensor], held: list):
    """Bridge a Python ``func(*tensors) -> Tensor`` to a C ``CMLTransformFn``."""

    @ffi.callback(_TRANSFORM_SIG)
    def cb(inputs, n, ud):  # noqa: ARG001 - ud unused
        args = [Tensor.borrow(inputs[i]) for i in range(n)]
        out = func(*args)
        held.append(out)  # keep alive until the C transform returns
        return out._tensor

    return cb


def _forward_callback(func: Callable[[Tensor], Tensor], held: list):
    """Bridge a Python ``func(x) -> Tensor`` to a C ``CMLForwardFn``."""

    @ffi.callback(_FORWARD_SIG)
    def cb(x, ctx):  # noqa: ARG001 - ctx unused
        out = func(Tensor.borrow(x))
        held.append(out)
        return out._tensor

    return cb


def _in_array(args: Sequence[Tensor]):
    return ffi.new("Tensor*[]", [a._tensor for a in args])


def grad(func: Callable[..., Tensor], argnums: Union[int, Tuple[int, ...]] = 0):
    """Return a function computing the gradient of scalar-output ``func`` with
    respect to the positional argument(s) selected by ``argnums`` (torch.func.grad)."""

    def wrapped(*args: Tensor):
        held: list = []
        cb = _transform_callback(func, held)
        inarr = _in_array(args)
        if isinstance(argnums, int):
            g = lib.cml_grad(cb, ffi.NULL, inarr, len(args), argnums)
            return _wrap(g)
        nums = list(argnums)
        out = ffi.new("Tensor*[]", len(nums))
        arr = ffi.new("int[]", nums)
        rc = lib.cml_grad_multi(cb, ffi.NULL, inarr, len(args), arr, len(nums), out)
        if rc != 0:
            raise RuntimeError("cml_grad_multi failed")
        return tuple(_wrap(out[i]) for i in range(len(nums)))

    return wrapped


def vjp(func: Callable[..., Tensor], *primals: Tensor):
    """Evaluate ``func(*primals)`` and return ``(output, vjp_fn)`` where
    ``vjp_fn(cotangent)`` gives the input gradients (torch.func.vjp)."""
    held: list = []
    cb = _transform_callback(func, held)
    inarr = _in_array(primals)
    out_pp = ffi.new("Tensor**")
    handle = lib.cml_vjp(cb, ffi.NULL, inarr, len(primals), out_pp)
    if handle == ffi.NULL:
        raise RuntimeError("cml_vjp failed")
    output = _wrap(out_pp[0])

    n = len(primals)
    # keep cb / inarr / handle referenced by the closure below
    def vjp_fn(cotangent: Tensor):
        grads = ffi.new("Tensor*[]", n)
        if lib.cml_vjp_fn(handle, cotangent._tensor, grads) != 0:
            raise RuntimeError("cml_vjp_fn failed")
        return tuple(_wrap(grads[i]) for i in range(n))

    vjp_fn._handle = handle  # type: ignore[attr-defined]
    vjp_fn._cb = cb  # type: ignore[attr-defined]
    return output, vjp_fn


def jvp(func: Callable[[Tensor], Tensor], primals, tangents):
    """Forward-mode Jacobian-vector product. ``primals``/``tangents`` are a
    single tensor or a 1-tuple of one (torch.func.jvp, single-input form).
    Requires the graph engine."""
    _require_graph("jvp")
    x = primals[0] if isinstance(primals, (tuple, list)) else primals
    v = tangents[0] if isinstance(tangents, (tuple, list)) else tangents
    held: list = []
    cb = _forward_callback(func, held)
    out = lib.cml_jvp(cb, ffi.NULL, x._tensor, v._tensor)
    if out == ffi.NULL:
        raise RuntimeError("cml_jvp returned NULL")
    return func(x), _wrap(out)


def jacrev(func: Callable[..., Tensor], argnums: int = 0):
    """Return a function computing the Jacobian of ``func`` via reverse mode
    (torch.func.jacrev). Shape ``[numel(output), numel(arg)]``."""

    def wrapped(*args: Tensor):
        held: list = []
        cb = _transform_callback(func, held)
        inarr = _in_array(args)
        j = lib.cml_jacrev(cb, ffi.NULL, inarr, len(args), argnums)
        if j == ffi.NULL:
            raise RuntimeError("cml_jacrev returned NULL")
        return _wrap(j)

    return wrapped


def jacfwd(func: Callable[[Tensor], Tensor], argnums: int = 0):
    """Return a function computing the Jacobian of single-input ``func`` via
    forward mode (torch.func.jacfwd). Requires the graph engine."""

    def wrapped(*args: Tensor):
        _require_graph("jacfwd")
        if argnums != 0 or len(args) != 1:
            raise ValueError("cml.func.jacfwd supports a single input (argnums=0)")
        held: list = []
        cb = _forward_callback(func, held)
        j = lib.cml_jacfwd(cb, ffi.NULL, args[0]._tensor)
        if j == ffi.NULL:
            raise RuntimeError("cml_jacfwd returned NULL")
        return _wrap(j)

    return wrapped


def hessian(func: Callable[..., Tensor], argnums: int = 0):
    """Return a function computing the Hessian of scalar-output ``func``
    (torch.func.hessian). Requires the graph engine."""

    def wrapped(*args: Tensor):
        _require_graph("hessian")
        held: list = []
        cb = _transform_callback(func, held)
        inarr = _in_array(args)
        h = lib.cml_hessian(cb, ffi.NULL, inarr, len(args), argnums)
        if h == ffi.NULL:
            raise RuntimeError("cml_hessian returned NULL")
        return _wrap(h)

    return wrapped


def vmap(
    func: Callable[..., Tensor],
    in_dims: Union[int, Sequence[int]] = 0,
    out_dims: int = 0,
):
    """Return a vectorizing map over ``func`` (torch.func.vmap). This is the
    reference (non-fused) implementation: it slices each mapped input, applies
    ``func`` per slice, and stacks the results. ``in_dims`` is an int applied to
    every input, or a per-input sequence (use a negative value / ``None`` to
    leave an input unmapped)."""

    def wrapped(*args: Tensor):
        held: list = []
        cb = _transform_callback(func, held)
        inarr = _in_array(args)
        if isinstance(in_dims, int):
            axes = [in_dims] * len(args)
        else:
            axes = [(-1 if d is None else d) for d in in_dims]
        axarr = ffi.new("int[]", axes)
        r = lib.cml_vmap(cb, ffi.NULL, inarr, len(args), axarr, out_dims)
        if r == ffi.NULL:
            raise RuntimeError("cml_vmap returned NULL")
        return _wrap(r)

    return wrapped


def functional_call(
    module,
    parameters: Dict[str, Tensor],
    args: Union[Tensor, Tuple[Tensor, ...]],
):
    """Run ``module``'s forward with ``parameters`` swapped in by name, then
    restore the module's own parameters (torch.func.functional_call). ``args`` is
    the single forward input (or a 1-tuple of it)."""
    inp = args[0] if isinstance(args, (tuple, list)) else args
    sd = lib.nn_state_dict_create()
    if sd == ffi.NULL:
        raise RuntimeError("nn_state_dict_create failed")
    try:
        for name, tensor in parameters.items():
            if lib.nn_state_dict_set(sd, name.encode(), tensor._tensor) != 0:
                raise RuntimeError(f"nn_state_dict_set failed for '{name}'")
        out = lib.cml_functional_call(module._as_module(), sd, inp._tensor)
        if out == ffi.NULL:
            raise RuntimeError("cml_functional_call returned NULL")
        return _wrap(out)
    finally:
        lib.nn_state_dict_free(sd)
