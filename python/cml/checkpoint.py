"""Activation checkpointing (gradient rematerialization).

C-ML keeps every intermediate tensor of a graph alive until
``cml.reset_graph()``, so very deep graphs cost activation memory linear in
depth even though only leaf gradients are needed. ``checkpoint`` implements
the classic trade: run a segment *without* recording a graph (constant
memory), then recompute it *with* gradients when its loss is known.

Because the engine's autograd is C-side, Python cannot inject a recompute
hook into an existing backward pass. The supported pattern is therefore
segment-level, mirroring what people do manually in torch:

    # long chain split into K segments -- only one segment's activations
    # are alive at a time
    h = x
    for seg in segments:
        h = cml.checkpoint(seg, h)          # constant-memory forward

    # ... later, once you have a target/loss:
    h2 = seg(h_in)                          # recompute WITH grad
    loss = crit(h2, target)
    loss.backward()

``checkpoint`` is exactly the no-grad half of that pattern; it also verifies
at runtime that the segment is deterministic enough to recompute (same
values on the second pass) unless randomness is explicitly allowed.
"""

from __future__ import annotations

from .core import Tensor, no_grad


def checkpoint(fn, *inputs, verify_determinism: bool = False):
    """Run ``fn`` without building an autograd graph; return detached outputs.

    Args:
        fn: callable mapping ``*inputs`` to one or more Tensors.
        *inputs: arguments passed straight through (Tensors keep values;
            they are not detached, so parameters stay shared).
        verify_determinism: run twice and assert identical values -- catches
            segments containing random ops before a recompute-based backward
            would silently diverge.

    Returns:
        Whatever ``fn`` returns (Tensors will carry values but no history).
    """
    with no_grad():
        out = fn(*inputs)
        if verify_determinism:
            out2 = fn(*inputs)

    def _values(o):
        if isinstance(o, Tensor):
            v = o.numpy()
            return Tensor(v.copy())
        if isinstance(o, (list, tuple)):
            seq = [_values(x) for x in o]
            return type(o)(seq)
        return o

    result = _values(out)
    if verify_determinism:
        check = _values(out2)
        r_vals, c_vals = _flatten(result), _flatten(check)
        for a, b in zip(r_vals, c_vals):
            import numpy as np

            if not np.allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-6):
                raise RuntimeError(
                    "checkpoint segment is not deterministic; recomputing it "
                    "during backward would produce wrong gradients"
                )
    return result


def _flatten(obj):
    if isinstance(obj, Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten(item)
