"""Core CML functionality."""

from __future__ import annotations
from typing import Union, Optional, Tuple
import operator
import numpy as np
from cml._cml_lib import ffi, lib


def _get_lib():
    """Return the loaded CFFI ``lib`` handle (compat accessor used by submodules)."""
    return lib


def _get_ffi():
    """Return the CFFI ``ffi`` instance."""
    return ffi


DEVICE_CPU = 0
DEVICE_CUDA = 1
DEVICE_METAL = 2
DEVICE_ROCM = 3
DEVICE_SIM_GPU = 4
DEVICE_AUTO = 5

DEVICE_NAMES = {
    DEVICE_CPU: "CPU",
    DEVICE_CUDA: "CUDA",
    DEVICE_METAL: "Metal",
    DEVICE_ROCM: "ROCm",
    DEVICE_SIM_GPU: "SimGPU",
    DEVICE_AUTO: "Auto",
}

DTYPE_FLOAT32 = 0
DTYPE_FLOAT64 = 1
DTYPE_INT32 = 2
DTYPE_INT64 = 3
DTYPE_BOOL = 4
DTYPE_FLOAT16 = 5
DTYPE_BFLOAT16 = 6
DTYPE_INT8 = 7
DTYPE_UINT8 = 8

DTYPE_NAMES = {
    DTYPE_FLOAT32: "float32",
    DTYPE_FLOAT64: "float64",
    DTYPE_INT32: "int32",
    DTYPE_INT64: "int64",
    DTYPE_BOOL: "bool",
    DTYPE_FLOAT16: "float16",
    DTYPE_BFLOAT16: "bfloat16",
    DTYPE_INT8: "int8",
    DTYPE_UINT8: "uint8",
}

DTYPE_TO_NUMPY = {
    DTYPE_FLOAT32: np.float32,
    DTYPE_FLOAT64: np.float64,
    DTYPE_INT32: np.int32,
    DTYPE_INT64: np.int64,
}

NUMPY_TO_DTYPE = {v: k for k, v in DTYPE_TO_NUMPY.items()}


def init():
    lib.cml_init()


def reset_graph():
    """Fully reset the global autograd/IR state and drop all execution caches.

    Training a model already resets its own step graph automatically. Call this
    only *between* training different models in the same process, after releasing
    the previous model/optimizer (e.g. ``del model, opt``), to clear cached
    execution plans that are keyed by tensor shape and would otherwise be
    replayed with the previous model's freed buffers. Do NOT call it mid-step.
    """
    lib.cml_reset_ir_context()


def cleanup():
    lib.cml_cleanup()


def seed(s):
    lib.cml_seed(int(s))
    np.random.seed(s)


def get_device():
    return lib.cml_get_default_device()


def set_device(device: int):
    try:
        lib.cml_set_default_device(device)
    except (AttributeError, Exception):
        if device != DEVICE_CPU:
            import warnings
            warnings.warn(f"Device {DEVICE_NAMES.get(device, device)} not available, using CPU")


def get_dtype():
    return lib.cml_get_default_dtype()


def set_dtype(dtype):
    lib.cml_set_default_dtype(dtype)


def _make_config(dtype=None, device=None):
    config = ffi.new("TensorConfig*")
    if dtype is not None:
        config.dtype = dtype
        config.has_dtype = True
    else:
        config.dtype = lib.cml_get_default_dtype()
        config.has_dtype = True
    if device is not None:
        config.device = device
        config.has_device = True
    else:
        config.has_device = False
    return config


def _coerce_shape(shape):
    if isinstance(shape, int):
        shape = [shape]
    return list(shape)


def _validate_shape(shape):
    shape = _coerce_shape(shape)
    if not shape:
        raise ValueError("shape must be non-empty")
    for i, dim in enumerate(shape):
        if not isinstance(dim, int):
            raise TypeError(f"shape[{i}] must be int, got {type(dim).__name__}")
        if dim < 0:
            raise ValueError(f"shape[{i}] must be non-negative, got {dim}")
    return shape


def _validate_dtype(dtype):
    if dtype is None:
        return DTYPE_FLOAT32
    if isinstance(dtype, str):
        name = dtype.lower()
        rev = {v: k for k, v in DTYPE_NAMES.items()}
        if name not in rev:
            raise ValueError(f"unsupported dtype string: {dtype!r}")
        return rev[name]
    if dtype not in DTYPE_TO_NUMPY:
        raise ValueError(f"unsupported dtype id: {dtype}")
    return dtype


def _create(lib_fn, shape, dtype, device, *extra_args):
    shape = _validate_shape(shape)
    dtype = _validate_dtype(dtype)
    shape_array = ffi.new("int[]", shape)
    config = _make_config(dtype, device)
    args = [shape_array, len(shape), config] + list(extra_args)
    return Tensor(lib_fn(*args))


def _cmp(self, other, np_op):
    if isinstance(other, (int, float)):
        return Tensor.from_numpy(np_op(self.numpy(), other).astype(np.float32))
    if isinstance(other, Tensor):
        return Tensor.from_numpy(np_op(self.numpy(), other.numpy()).astype(np.float32))
    return NotImplemented


def tensor(data, dtype=None, device=None, requires_grad=False) -> "Tensor":
    """PyTorch/numpy-style tensor factory.

    ``cml.tensor([[1, 2], [3, 4]])`` builds a tensor from (nested) lists, a
    scalar, a tuple, or a numpy array.
    """
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def einsum(equation: str, *operands) -> "Tensor":
    """NumPy-style ``einsum("ij,jk->ik", a, b)`` over Tensor operands."""
    ops = [o if isinstance(o, Tensor) else Tensor(o) for o in operands]
    arr = ffi.new("Tensor*[]", [o._tensor for o in ops])
    out = lib.cml_einsum(equation.encode(), arr, len(ops))
    if out == ffi.NULL:
        raise ValueError(f"einsum: invalid equation {equation!r} or operand shapes")
    return Tensor(out)


def where(condition, x, y) -> "Tensor":
    """NumPy/torch-style ``where(cond, x, y)``."""
    ct = condition if isinstance(condition, Tensor) \
        else Tensor(np.asarray(condition, dtype=np.float32))
    xt = x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=np.float32))
    yt = y if isinstance(y, Tensor) else Tensor(np.asarray(y, dtype=np.float32))
    return Tensor(lib.cml_where(ct._tensor, xt._tensor, yt._tensor))


# Type-promotion lattice (torch semantics, restricted to the dtypes the
# bindings can represent). Higher rank wins within a category; a floating
# operand always wins over an integral one. All ranks here are distinct, so
# "wider" is simply the higher rank.
_PROMOTE_RANK = {
    DTYPE_BOOL: 0,
    DTYPE_INT8: 1,
    DTYPE_UINT8: 2,
    DTYPE_INT32: 3,
    DTYPE_INT64: 4,
    DTYPE_FLOAT16: 5,
    DTYPE_BFLOAT16: 6,
    DTYPE_FLOAT32: 7,
    DTYPE_FLOAT64: 8,
}


def result_type(a, b) -> int:
    """DType produced by a binary op mixing dtypes ``a`` and ``b`` (torch
    ``torch.result_type``). Accepts dtype ids or Tensors."""
    da = a.dtype if isinstance(a, Tensor) else a
    db = b.dtype if isinstance(b, Tensor) else b
    ra = _PROMOTE_RANK.get(da)
    rb = _PROMOTE_RANK.get(db)
    if ra is None or rb is None:
        raise ValueError(f"result_type: unsupported dtype(s): {da}, {db}")
    return da if ra >= rb else db


def pad(x, pad_widths, mode: str = "constant", value: float = 0.0) -> "Tensor":
    """Pad a tensor (``torch.nn.functional.pad``-style).

    Accepts either an int (symmetric padding on every dim), a flat sequence of
    2*ndim ints interpreted as ``(before, after)`` pairs starting from the LAST
    dim (torch order), or a nested sequence of per-dim ``(before, after)``
    pairs in dim order (numpy order).
    """
    t = x if isinstance(x, Tensor) else Tensor(x)
    return t.pad(pad_widths, mode=mode, value=value)


class Tensor:
    _shape_cache: Optional[Tuple[int, ...]] = None

    def __init__(self, data=None, dtype=None, device=None, requires_grad=False):
        """Create a tensor.

        Accepts either raw data (a Python scalar / (nested) list / tuple /
        numpy array) — the ergonomic, PyTorch-style path — or, internally, an
        existing C tensor handle (a CFFI cdata pointer) which is simply wrapped.
        """
        self._shape_cache = None
        self._borrowed = False

        # Internal fast path: wrap an existing C tensor pointer as-is.
        if isinstance(data, ffi.CData):
            self._tensor = data
            # Register external ownership so a graph teardown (e.g. the per-step
            # reset inside optimizer.step()) detaches this tensor instead of
            # freeing it out from under this wrapper. Released in __del__.
            if data != ffi.NULL:
                lib.tensor_pin(data)
            return
        if data is None:
            self._tensor = ffi.NULL
            return

        # User path: build from Python / numpy data.
        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        if arr.ndim == 0:                    # scalar -> 1-element 1-D tensor
            arr = arr.reshape(1)
        built = Tensor.from_numpy(arr, requires_grad=requires_grad, dtype=dtype)
        # Take ownership of the built C tensor; neutralize the temporary so its
        # __del__ doesn't free the handle we just adopted.
        self._tensor = built._tensor
        built._tensor = ffi.NULL
        if device is not None:
            moved = self.to(device=device)
            if moved is not self:
                self._tensor, moved._tensor = moved._tensor, self._tensor

    @classmethod
    def borrow(cls, c_tensor):
        """Wrap a C tensor handle this Python side does NOT own (e.g. a module
        parameter owned by its Module). No pin is taken and __del__ never
        releases, so repeatedly wrapping the same handle cannot shorten the
        owner's lifetime."""
        obj = cls.__new__(cls)
        obj._shape_cache = None
        obj._borrowed = True
        obj._tensor = c_tensor if c_tensor is not None else ffi.NULL
        return obj

    def __del__(self):
        if (
            not getattr(self, "_borrowed", False)
            and hasattr(self, "_tensor")
            and self._tensor is not None
            and self._tensor != ffi.NULL
        ):
            # Drop our external reference; frees the tensor unless the C core
            # still holds it (mirrors the tensor_pin() taken when wrapping).
            lib.tensor_release(self._tensor)

    @staticmethod
    def _as_operand(other, dtype=None):
        """Coerce a binary-op operand to a Tensor (scalars become a 1-element
        tensor that the C ops broadcast), or return None for unsupported types.

        Python scalars are weakly typed (torch rule): they adopt the tensor's
        dtype instead of widening it — except a float scalar meeting an
        integral tensor, which promotes to float32 like torch's default
        floating dtype.
        """
        if isinstance(other, Tensor):
            return other
        if isinstance(other, (bool, int, float)):
            if dtype is None:
                dtype = DTYPE_FLOAT32
            elif isinstance(other, float) and dtype not in (DTYPE_FLOAT32, DTYPE_FLOAT64):
                dtype = DTYPE_FLOAT32
            if dtype in DTYPE_TO_NUMPY:
                return Tensor.from_numpy(
                    np.array([other], dtype=np.dtype(DTYPE_NAMES[dtype])),
                    dtype=dtype)
            # dtype outside the numpy round-trip table: build f32, then cast
            return Tensor.from_numpy(
                np.array([float(other)], dtype=np.float32)).cast(dtype)
        return None

    def _binary(self, other, lib_fn):
        """Shared binary-op path. Promotes both operands to one dtype before
        dispatching (the C engine does not promote mixed-dtype inputs).
        Python scalars are weakly typed: they adopt the tensor's dtype,
        except a float scalar meeting an integral tensor, which widens the
        result to float32 (torch's default-floating-dtype rule)."""
        if isinstance(other, Tensor):
            dt = result_type(self.dtype, other.dtype)
            a = self if self.dtype == dt else self.cast(dt)
            b = other if other.dtype == dt else other.cast(dt)
        elif isinstance(other, (bool, int, float)):
            dt = self.dtype
            if isinstance(other, float) and dt not in (DTYPE_FLOAT32, DTYPE_FLOAT64):
                dt = DTYPE_FLOAT32
            a = self if self.dtype == dt else self.cast(dt)
            b = Tensor._as_operand(other, dt)
        else:
            return NotImplemented
        return Tensor(lib_fn(a._tensor, b._tensor))

    def __add__(self, other):
        return self._binary(other, lib.cml_add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary(other, lib.cml_sub)

    def __rsub__(self, other):
        return self._rbin(other, lib.cml_sub)

    def __mul__(self, other):
        return self._binary(other, lib.cml_mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary(other, lib.cml_div)

    def __rtruediv__(self, other):
        return self._rbin(other, lib.cml_div)

    def _rbin(self, other, lib_fn):
        """Reflected binary op with the tensor on the right: computes
        other <op> self under the same promotion/weak-scalar rules."""
        if isinstance(other, Tensor):
            dt = result_type(self.dtype, other.dtype)
            a = other if other.dtype == dt else other.cast(dt)
            b = self if self.dtype == dt else self.cast(dt)
        elif isinstance(other, (bool, int, float)):
            dt = self.dtype
            if isinstance(other, float) and dt not in (DTYPE_FLOAT32, DTYPE_FLOAT64):
                dt = DTYPE_FLOAT32
            a = Tensor._as_operand(other, dt)
            b = self if self.dtype == dt else self.cast(dt)
        else:
            return NotImplemented
        return Tensor(lib_fn(a._tensor, b._tensor))

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(lib.cml_matmul(self._tensor, other._tensor))
        return NotImplemented

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._shape_cache is not None:
            return self._shape_cache
        if self._tensor is None or self._tensor == ffi.NULL:
            return ()
        ndim = self._tensor.ndim
        if ndim == 0:
            self._shape_cache = ()
            return ()
        shape_ptr = self._tensor.shape
        if shape_ptr == ffi.NULL:
            return ()
        self._shape_cache = tuple(shape_ptr[i] for i in range(ndim))
        return self._shape_cache

    @property
    def ndim(self) -> int:
        if self._tensor is None or self._tensor == ffi.NULL:
            return 0
        return self._tensor.ndim

    @property
    def numel(self) -> int:
        if self._tensor is None or self._tensor == ffi.NULL:
            return 0
        return self._tensor.numel

    @property
    def size(self) -> int:
        return self.numel

    @property
    def dtype(self) -> int:
        if self._tensor is None or self._tensor == ffi.NULL:
            return DTYPE_FLOAT32
        return self._tensor.dtype

    @property
    def device(self) -> int:
        if self._tensor is None or self._tensor == ffi.NULL:
            return DEVICE_CPU
        return self._tensor.device

    @property
    def requires_grad(self) -> bool:
        if self._tensor is None or self._tensor == ffi.NULL:
            return False
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        if self._tensor is not None and self._tensor != ffi.NULL:
            lib.cml_set_requires_grad(self._tensor, value)

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        self.requires_grad = requires_grad
        return self

    @property
    def grad(self) -> Optional["Tensor"]:
        if self._tensor is None or self._tensor == ffi.NULL:
            return None
        grad_ptr = self._tensor.grad
        if grad_ptr == ffi.NULL:
            return None
        return _TensorView(grad_ptr)

    @property
    def is_contiguous(self) -> bool:
        if self._tensor is None or self._tensor == ffi.NULL:
            return True
        return lib.tensor_is_contiguous(self._tensor)

    def is_scalar(self) -> bool:
        if self._tensor is None or self._tensor == ffi.NULL:
            return False
        return lib.tensor_is_scalar(self._tensor)

    def item(self) -> float:
        if self.numel != 1:
            raise ValueError(
                f"only one element tensors can be converted to Python scalars, got {self.numel} elements"
            )
        return lib.tensor_get_float(self._tensor, 0)

    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = new_shape[0]
        shape_array = ffi.new("int[]", new_shape)
        return Tensor(lib.cml_reshape(self._tensor, shape_array, len(new_shape)))

    def view(self, *new_shape) -> "Tensor":
        return self.reshape(*new_shape)

    def slice(self, start: int, stop: int) -> "Tensor":
        """Contiguous slice [start, stop) along dim 0, like tensor[start:stop]."""
        nd = self.ndim
        shape = self.shape
        if nd < 1:
            raise ValueError("slice requires at least a 1-D tensor")
        n = shape[0]
        if start < 0:
            start += n
        if stop < 0:
            stop += n
        start = max(0, min(start, n))
        stop = max(start, min(stop, n))
        starts = ffi.new("int[]", [start] + [0] * (nd - 1))
        ends = ffi.new("int[]", [stop] + list(shape[1:]))
        return Tensor(lib.uop_shrink(self._tensor, starts, ends, nd))

    def index_select(self, indices) -> "Tensor":
        """Rows at the given indices along dim 0 (lazy gather)."""
        idx = Tensor([float(i) for i in indices])
        return Tensor(lib.uop_gather(self._tensor, idx._tensor, 0))

    def transpose(self, dim0=0, dim1=1):
        return Tensor(lib.cml_transpose(self._tensor, dim0, dim1))

    @property
    def T(self) -> "Tensor":
        return self.transpose(0, 1)

    def sum(self, dim=-1, keepdim=False):
        return Tensor(lib.cml_sum(self._tensor, dim, keepdim))

    def mean(self, dim=-1, keepdim=False):
        return Tensor(lib.cml_mean(self._tensor, dim, keepdim))

    def max(self, dim=-1, keepdim=False):
        return Tensor(lib.cml_max(self._tensor, dim, keepdim))

    def min(self, dim=-1, keepdim=False):
        return Tensor(lib.cml_min(self._tensor, dim, keepdim))

    def prod(self, dim=-1, keepdim=False):
        return Tensor(lib.cml_prod(self._tensor, dim, keepdim))

    def argmax(self, dim=-1):
        return Tensor(lib.cml_argmax(self._tensor, dim))

    def argmin(self, dim=-1):
        return Tensor(lib.cml_argmin(self._tensor, dim))

    def var(self, dim=-1, unbiased=True, keepdim=False):
        return Tensor(lib.cml_var(self._tensor, dim, unbiased, keepdim))

    def std(self, dim=-1, unbiased=True, keepdim=False):
        return Tensor(lib.cml_std(self._tensor, dim, unbiased, keepdim))

    def softmax(self, dim=1):
        return Tensor(lib.cml_softmax(self._tensor, dim))

    def pow(self, other):
        if isinstance(other, Tensor):
            return Tensor(lib.cml_pow(self._tensor, other._tensor))
        return self.__pow__(other)  # scalar exponent, like torch.Tensor.pow

    def clamp(self, min_val, max_val):
        return Tensor(lib.cml_clamp(self._tensor, float(min_val), float(max_val)))

    def clone(self):
        return Tensor(lib.cml_clone(self._tensor))

    def detach(self):
        return Tensor(lib.cml_detach(self._tensor))

    def contiguous(self) -> "Tensor":
        if self.is_contiguous:
            return self
        return Tensor(lib.cml_contiguous(self._tensor))

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        if dim is None:
            dim = -1
        return Tensor(lib.cml_squeeze(self._tensor, dim))

    def unsqueeze(self, dim: int) -> "Tensor":
        return Tensor(lib.cml_unsqueeze(self._tensor, dim))

    def flip(self, dim: int) -> "Tensor":
        return Tensor(lib.cml_flip(self._tensor, dim))

    def sort(self, dim: int = -1, descending: bool = False) -> "Tensor":
        return Tensor(lib.cml_sort(self._tensor, dim, descending))

    def cast(self, dtype: int) -> "Tensor":
        return Tensor(lib.cml_cast(self._tensor, dtype))

    def dot(self, other: "Tensor") -> "Tensor":
        return Tensor(lib.cml_dot(self._tensor, other._tensor))

    def matmul(self, other: "Tensor") -> "Tensor":
        return Tensor(lib.cml_matmul(self._tensor, other._tensor))

    def where(self, condition, other):
        """torch-style ``self.where(cond, other)``: cond ? self : other."""
        if not isinstance(condition, Tensor):
            condition = Tensor(np.asarray(condition, dtype=np.float32))
        if not isinstance(other, Tensor):
            other = Tensor(np.asarray(other, dtype=np.float32))
        return Tensor(lib.cml_where(condition._tensor, self._tensor, other._tensor))

    def roll(self, shift: int, axis: int = 0) -> "Tensor":
        """NumPy-style circular shift along ``axis``."""
        return Tensor(lib.cml_roll(self._tensor, int(shift), int(axis)))

    def copysign(self, other) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(np.asarray(other, dtype=np.float32))
        return Tensor(lib.cml_copysign(self._tensor, other._tensor))

    def logaddexp(self, other) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(np.asarray(other, dtype=np.float32))
        return Tensor(lib.cml_logaddexp(self._tensor, other._tensor))

    def one_hot(self, num_classes: int) -> "Tensor":
        """``self`` holds integer class indices; returns one-hot encoding with
        a new trailing axis of size ``num_classes``."""
        return Tensor(lib.cml_one_hot(self._tensor, int(num_classes)))

    def cumsum(self, dim: int = -1) -> "Tensor":
        return Tensor(lib.cml_cumsum(self._tensor, dim))

    def cumprod(self, dim: int = -1) -> "Tensor":
        return Tensor(lib.cml_cumprod(self._tensor, dim))

    def logcumsumexp(self, dim: int = -1) -> "Tensor":
        return Tensor(lib.cml_logcumsumexp(self._tensor, dim))

    def argsort(self, dim: int = -1, descending: bool = False) -> "Tensor":
        return Tensor(lib.cml_argsort(self._tensor, dim, descending))

    def topk(self, k: int, dim: int = -1, largest: bool = True,
             sorted: bool = True) -> Tuple["Tensor", "Tensor"]:
        """torch-style top-k: returns ``(values, indices)``."""
        idx_ptr = ffi.new("Tensor**")
        values = lib.cml_topk_with_indices(self._tensor, int(k), dim, largest, idx_ptr)
        if values == ffi.NULL:
            raise ValueError(f"topk: invalid k={k} or dim={dim} for shape {self.shape}")
        indices = Tensor(idx_ptr[0]) if idx_ptr[0] != ffi.NULL else None
        return Tensor(values), indices

    def masked_select(self, mask) -> "Tensor":
        """Select elements where ``mask`` (broadcastable bool tensor/array) is
        true; returns a flat 1-D tensor (torch.masked_select semantics)."""
        if not isinstance(mask, Tensor):
            mask = Tensor(np.asarray(mask, dtype=np.float32))
        return Tensor(lib.cml_masked_select(self._tensor, mask._tensor))

    def median(self, dim: Optional[int] = None):
        """Median along ``dim``, or the global median when ``dim`` is None.
        numpy semantics: for even counts the two central values are averaged
        (``torch.median`` instead returns the lower one)."""
        n = self.numel
        if n == 0:
            raise ValueError("median of an empty tensor")
        if dim is None:
            s = self.flatten().sort(-1)
            if n % 2:
                v = lib.tensor_get_float(s._tensor, n // 2)
            else:
                v = 0.5 * (lib.tensor_get_float(s._tensor, n // 2 - 1)
                           + lib.tensor_get_float(s._tensor, n // 2))
            return Tensor.from_numpy(
                np.array([v], dtype=np.float32))
        shp = self.shape
        d = dim + len(shp) if dim < 0 else dim
        cnt = shp[d]
        s = self.sort(d)
        mid = cnt // 2
        if cnt % 2:
            out = s._gather_indices([mid], d)
        else:
            out = s._gather_indices([mid - 1, mid], d).mean(d)
        return out.squeeze(d)

    def unique(self) -> "Tensor":
        """Sorted unique values of ``self`` (eager composition over numpy,
        like ``log10``)."""
        u = np.unique(self.numpy())
        return Tensor.from_numpy(u.astype(np.float32))

    def pad(self, pad_widths, mode: str = "constant", value: float = 0.0) -> "Tensor":
        """Pad this tensor. See module-level ``cml.pad`` for the accepted
        ``pad_widths`` forms; ``mode`` is one of ``constant``/``reflect``/
        ``replicate`` (``edge`` is accepted as an alias of ``replicate``,
        ``symmetric`` is not supported)."""
        nd = self.ndim
        if nd < 1:
            raise ValueError("pad requires at least a 1-D tensor")
        if isinstance(pad_widths, (int, np.integer)):
            pairs = [(int(pad_widths), int(pad_widths))] * nd
        elif isinstance(pad_widths, (list, tuple)) and len(pad_widths) > 0 \
                and isinstance(pad_widths[0], (list, tuple)):
            # nested per-dim (before, after), dim order
            if len(pad_widths) != nd:
                raise ValueError(f"pad: expected {nd} (before, after) pairs, "
                                 f"got {len(pad_widths)}")
            pairs = [(int(p[0]), int(p[1])) for p in pad_widths]
        else:
            # flat sequence of ints, torch order: last dim first. Fewer than
            # 2*nd ints pad only the trailing dims (leading dims untouched),
            # like torch.nn.functional.pad.
            flat = [int(w) for w in pad_widths]
            if len(flat) % 2 != 0 or len(flat) > 2 * nd:
                raise ValueError(f"pad: expected up to {2 * nd} ints (even count) "
                                 f"for a {nd}-D tensor, got {len(flat)}")
            tail = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)][::-1]
            pairs = [(0, 0)] * (nd - len(tail)) + tail
        widths = ffi.new("int[]", [w for p in pairs for w in p])
        m = mode.lower()
        if m == "constant":
            return Tensor(lib.cml_pad(self._tensor, widths, nd, float(value)))
        if m == "reflect":
            return Tensor(lib.cml_pad_reflect(self._tensor, widths, nd))
        if m in ("replicate", "edge"):
            return Tensor(lib.cml_pad_replicate(self._tensor, widths, nd))
        raise ValueError(f"pad: unsupported mode {mode!r}")

    def relu(self): return Tensor(lib.cml_relu(self._tensor))
    def sigmoid(self): return Tensor(lib.cml_sigmoid(self._tensor))
    def tanh(self): return Tensor(lib.cml_tanh(self._tensor))
    def exp(self): return Tensor(lib.cml_exp(self._tensor))
    def log(self): return Tensor(lib.cml_log(self._tensor))
    def sqrt(self): return Tensor(lib.cml_sqrt(self._tensor))
    def sin(self): return Tensor(lib.cml_sin(self._tensor))
    def cos(self): return Tensor(lib.cml_cos(self._tensor))
    def log2(self) -> "Tensor": return Tensor(lib.cml_log2(self._tensor))
    def tan(self) -> "Tensor": return Tensor(lib.cml_tan(self._tensor))
    def asin(self) -> "Tensor": return Tensor(lib.cml_asin(self._tensor))
    def acos(self) -> "Tensor": return Tensor(lib.cml_acos(self._tensor))
    def atan(self) -> "Tensor": return Tensor(lib.cml_atan(self._tensor))
    def rsqrt(self) -> "Tensor": return Tensor(lib.cml_rsqrt(self._tensor))
    def erf(self) -> "Tensor": return Tensor(lib.cml_erf(self._tensor))
    def exp2(self) -> "Tensor": return Tensor(lib.cml_exp2(self._tensor))
    def sign(self) -> "Tensor": return Tensor(lib.cml_sign(self._tensor))
    def ceil(self) -> "Tensor": return Tensor(lib.cml_ceil(self._tensor))
    def floor(self) -> "Tensor": return Tensor(lib.cml_floor(self._tensor))
    def square(self) -> "Tensor": return Tensor(lib.cml_square(self._tensor))
    def round(self, decimals: int = 0) -> "Tensor": return Tensor(lib.cml_round(self._tensor))

    def log10(self) -> "Tensor":
        return Tensor.from_numpy(np.log10(self.numpy()))

    def reciprocal(self) -> "Tensor":
        ones = Tensor.full(self.shape or [self.size], 1.0)
        return Tensor(lib.cml_div(ones._tensor, self._tensor))

    def numpy(self) -> np.ndarray:
        if self._tensor is None or self._tensor == ffi.NULL:
            raise RuntimeError("Cannot convert null tensor to numpy")

        lib.tensor_ensure_executed(self._tensor)

        data_ptr = lib.tensor_data_ptr(self._tensor)
        if data_ptr == ffi.NULL:
            raise RuntimeError("Tensor data is null after execution")

        numel = self.numel
        shape = self.shape
        np_dtype = DTYPE_TO_NUMPY.get(self.dtype, np.float32)
        dtype_size = np.dtype(np_dtype).itemsize

        buffer = ffi.buffer(data_ptr, numel * dtype_size)
        arr = np.frombuffer(buffer, dtype=np_dtype).copy()

        if shape:
            arr = arr.reshape(shape)
        return arr

    def __array__(self, dtype=None) -> np.ndarray:
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_numpy(cls, arr: np.ndarray, requires_grad: bool = False,
                   dtype: Optional[int] = None) -> "Tensor":
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"from_numpy expects numpy.ndarray, got {type(arr).__name__}")

        target_dtype = _validate_dtype(dtype) if dtype is not None else DTYPE_FLOAT32
        np_target = DTYPE_TO_NUMPY[target_dtype]

        if arr.dtype != np_target:
            arr = arr.astype(np_target)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        shape = _validate_shape(arr.shape if arr.shape else (1,))
        if arr.size == 0:
            raise ValueError("from_numpy: array must contain at least one element")

        expected_numel = 1
        for d in shape:
            expected_numel *= d
        if arr.size != expected_numel:
            raise ValueError(
                f"from_numpy: array size {arr.size} does not match shape {tuple(shape)} "
                f"(expected {expected_numel} elements)"
            )

        shape_array = ffi.new("int[]", shape)
        data_ptr = ffi.cast("void*", arr.ctypes.data)
        # Pass the dtype through — a NULL config defaults to float32, which
        # reinterprets non-f32 array bytes as float garbage.
        config = ffi.new("TensorConfig*", {"dtype": target_dtype, "has_dtype": True})
        c_tensor = lib.tensor_from_data(data_ptr, shape_array, len(shape), config)
        if c_tensor == ffi.NULL:
            raise RuntimeError("Failed to create tensor from numpy array")

        tensor = cls(c_tensor)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor

    def to(self, device: Union[int, str] = None, dtype: Union[int, str] = None) -> "Tensor":
        if device is not None:
            dev = device.lower() if isinstance(device, str) else device
            if dev not in (DEVICE_CPU, "cpu"):
                raise NotImplementedError(
                    f"Tensor.to: only CPU tensors are supported, got device={device!r}"
                )
        if dtype is not None:
            return self.cast(dtype)
        return self.clone()

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        grad_ptr = gradient._tensor if gradient is not None else ffi.NULL
        lib.cml_backward(self._tensor, grad_ptr, retain_graph, create_graph)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        shape = list(self.shape)
        if end_dim < 0:
            end_dim = len(shape) + end_dim
        if start_dim < 0:
            start_dim = len(shape) + start_dim
        flat_size = 1
        for i in range(start_dim, end_dim + 1):
            flat_size *= shape[i]
        new_shape = shape[:start_dim] + [flat_size] + shape[end_dim + 1:]
        return self.reshape(new_shape)

    def __neg__(self) -> "Tensor":
        neg_one = Tensor.full(self.shape, -1.0)
        return self * neg_one

    def __pos__(self) -> "Tensor":
        return self

    def __abs__(self) -> "Tensor":
        return self.relu() + (-self).relu()

    def __eq__(self, other) -> bool:
        if isinstance(other, Tensor):
            return self._tensor == other._tensor
        return False

    def __hash__(self) -> int:
        return hash(int(ffi.cast("uintptr_t", self._tensor)))

    def __lt__(self, other): return _cmp(self, other, operator.lt)
    def __gt__(self, other): return _cmp(self, other, operator.gt)
    def __le__(self, other): return _cmp(self, other, operator.le)
    def __ge__(self, other): return _cmp(self, other, operator.ge)
    def __ne__(self, other): return _cmp(self, other, operator.ne)

    def __pow__(self, power: Union[int, float]) -> "Tensor":
        power_tensor = Tensor.full(self.shape or [self.size], float(power))
        return Tensor(lib.cml_pow(self._tensor, power_tensor._tensor))

    def __mod__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            other_t = Tensor.full(self.shape or [self.size], float(other))
            return Tensor(lib.cml_mod(self._tensor, other_t._tensor))
        if isinstance(other, Tensor):
            return Tensor(lib.cml_mod(self._tensor, other._tensor))
        raise TypeError(f"Cannot compute modulo of Tensor and {type(other)}")

    def __floordiv__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            other_t = Tensor.full(self.shape or [self.size], float(other))
            result = lib.cml_div(self._tensor, other_t._tensor)
            return Tensor(lib.cml_floor(result))
        if isinstance(other, Tensor):
            result = lib.cml_div(self._tensor, other._tensor)
            return Tensor(lib.cml_floor(result))
        raise TypeError("Floor division with tensors not supported")

    def __repr__(self) -> str:
        shape_str = str(self.shape) if self.shape else "()"
        dtype_str = DTYPE_NAMES.get(self.dtype, "unknown")
        device_str = DEVICE_NAMES.get(self.device, "unknown")
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={shape_str}, dtype={dtype_str}, device={device_str}{grad_str})"

    def __str__(self) -> str:
        try:
            return f"Tensor({self.numpy()})"
        except Exception:
            return repr(self)

    def __len__(self) -> int:
        shape = self.shape
        if not shape:
            raise TypeError("len() of unsized tensor")
        return shape[0]

    def _shrink_dim(self, dim: int, start: int, stop: int) -> "Tensor":
        """Slice [start, stop) along ``dim`` (lazy, single shrink node)."""
        shp = self.shape
        starts = [0] * len(shp)
        ends = list(shp)
        starts[dim] = start
        ends[dim] = stop
        return Tensor(lib.uop_shrink(self._tensor,
                                     ffi.new("int[]", starts),
                                     ffi.new("int[]", ends), len(shp)))

    def _gather_indices(self, indices, dim: int) -> "Tensor":
        """Gather along ``dim`` with an integer index array (lazy gather).
        uop_gather follows dim-0 shape semantics (index.shape + trailing
        dims), so for inner dims we bubble ``dim`` to the front with adjacent
        transposes (order-preserving), gather, and bubble it back."""
        idx = Tensor.from_numpy(
            np.asarray(indices, dtype=np.float32).reshape(-1))
        if dim == 0:
            return Tensor(lib.uop_gather(self._tensor, idx._tensor, 0))
        t = self
        for ax in range(dim, 0, -1):
            t = t.transpose(ax - 1, ax)
        g = Tensor(lib.uop_gather(t._tensor, idx._tensor, 0))
        for ax in range(1, dim + 1):
            g = g.transpose(ax - 1, ax)
        return g

    def __getitem__(self, idx):
        if isinstance(idx, int):
            shape = self.shape
            if not shape:
                raise IndexError("0-d tensor cannot be indexed")
            if idx < 0:
                idx = shape[0] + idx
            if idx < 0 or idx >= shape[0]:
                raise IndexError(
                    f"index {idx} out of range for dimension 0 with size {shape[0]}"
                )
            if len(shape) == 1:
                return lib.tensor_get_float(self._tensor, idx)
            return self._shrink_dim(0, idx, idx + 1).squeeze(0)
        elif isinstance(idx, tuple):
            shape = self.shape
            if (len(idx) == len(shape)
                    and all(isinstance(i, int) and not isinstance(i, bool) for i in idx)):
                flat_idx = 0
                stride = 1
                for i in range(len(shape) - 1, -1, -1):
                    dim_idx = idx[i]
                    if dim_idx < 0:
                        dim_idx = shape[i] + dim_idx
                    if dim_idx < 0 or dim_idx >= shape[i]:
                        raise IndexError(
                            f"index {idx[i]} out of range for dimension {i} "
                            f"with size {shape[i]}"
                        )
                    flat_idx += dim_idx * stride
                    stride *= shape[i]
                return lib.tensor_get_float(self._tensor, flat_idx)

        # General path: normalize to a tuple of per-dim items.
        items = idx if isinstance(idx, tuple) else (idx,)
        nd = self.ndim

        # Expand Ellipsis into enough full slices to fill the remaining dims.
        n_ellipsis = sum(1 for it in items if it is Ellipsis)
        if n_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if n_ellipsis:
            pos = items.index(Ellipsis)
            fill = (slice(None),) * (nd - (len(items) - 1))
            items = items[:pos] + fill + items[pos + 1:]

        t = self
        dim = 0
        for item in items:
            if item is None:
                t = t.unsqueeze(dim)
                continue  # newaxis does not consume a source dim
            shp = t.shape
            if dim >= len(shp):
                raise IndexError(
                    f"too many indices for tensor of dimension {len(shp)}")
            size = shp[dim]

            if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                i = int(item)
                if i < 0:
                    i += size
                if i < 0 or i >= size:
                    raise IndexError(
                        f"index {item} out of range for dimension {dim} with size {size}")
                t = t._shrink_dim(dim, i, i + 1).squeeze(dim)
                continue  # consumed the dim

            if isinstance(item, slice):
                if item.step not in (None, 1):
                    return Tensor.from_numpy(self.numpy()[idx])
                s, e, _ = slice(item.start, item.stop, 1).indices(size)
                t = t._shrink_dim(dim, s, e)
                dim += 1
                continue

            # list / ndarray / Tensor: integer-array or boolean-mask index.
            if isinstance(item, (list, np.ndarray, Tensor)):
                if isinstance(item, Tensor):
                    arr = item.numpy()
                else:
                    arr = np.asarray(item)
                # Comparisons currently yield f32 0/1 tensors — accept them
                # (and any all-0/1 array) as boolean masks.
                is_mask = arr.dtype == bool or (
                    arr.dtype.kind == "f" and arr.size > 0
                    and bool(((arr == 0) | (arr == 1)).all()))
                if is_mask:
                    if arr.ndim == len(t.shape) - dim:
                        # Full-rank mask over the remaining dims: flattened
                        # selection (torch/numpy semantics).
                        mt = item if isinstance(item, Tensor) else \
                            Tensor.from_numpy(arr.astype(np.float32))
                        return Tensor(lib.cml_masked_select(t._tensor, mt._tensor))
                    sel = np.nonzero(arr.astype(bool))[0]
                    t = t._gather_indices(sel, dim)
                    dim += 1
                    continue
                elif arr.ndim == 0:
                    i = int(arr)
                    if i < 0:
                        i += size
                    if i < 0 or i >= size:
                        raise IndexError(f"index {arr} out of range for dimension {dim}")
                    t = t._shrink_dim(dim, i, i + 1).squeeze(dim)
                    continue
                elif arr.ndim >= 1:
                    sel = arr.astype(np.int64)
                    if np.any((sel < -size) | (sel >= size)):
                        raise IndexError(
                            f"index out of range for dimension {dim} with size {size}")
                    sel = np.where(sel < 0, sel + size, sel)
                    # uop_gather takes flat 1-D indices; restore the index
                    # array's shape on the output (torch advanced-indexing
                    # semantics: result = index.shape + remaining dims).
                    rest = tuple(t.shape[dim + 1:])
                    t = t._gather_indices(sel.reshape(-1), dim)
                    want = tuple(int(v) for v in arr.shape) + rest
                    if dim == 0 and t.shape != want:
                        t = t.reshape(want)
                    elif dim > 0:
                        want = tuple(t.shape[:dim]) + \
                            tuple(int(v) for v in arr.shape) + rest
                        if t.shape != want:
                            t = t.reshape(want)
                    dim += 1
                    continue

            return Tensor.from_numpy(self.numpy()[idx])

        return t

    def __setitem__(self, idx, value: float):
        if isinstance(idx, int):
            shape = self.shape
            if not shape:
                raise IndexError("0-d tensor cannot be indexed")
            if idx < 0:
                idx = shape[0] + idx
            if idx < 0 or idx >= shape[0]:
                raise IndexError(
                    f"index {idx} out of range for dimension 0 with size {shape[0]}"
                )
            if len(shape) == 1:
                lib.tensor_set_float(self._tensor, idx, float(value))
            else:
                row_size = 1
                for d in range(1, len(shape)):
                    row_size *= shape[d]
                flat_start = idx * row_size
                if isinstance(value, (int, float)):
                    for j in range(row_size):
                        lib.tensor_set_float(self._tensor, flat_start + j, float(value))
                elif hasattr(value, '_tensor'):
                    lib.tensor_ensure_executed(value._tensor)
                    for j in range(row_size):
                        v = lib.tensor_get_float(value._tensor, j)
                        lib.tensor_set_float(self._tensor, flat_start + j, v)
                else:
                    raise TypeError(f"Cannot set tensor elements from {type(value)}")
        elif isinstance(idx, tuple):
            shape = self.shape
            if len(idx) != len(shape):
                raise IndexError(f"too many indices for tensor of dimension {len(shape)}")
            flat_idx = 0
            stride = 1
            for i in range(len(shape) - 1, -1, -1):
                dim_idx = idx[i]
                if dim_idx < 0:
                    dim_idx = shape[i] + dim_idx
                if dim_idx < 0 or dim_idx >= shape[i]:
                    raise IndexError(
                        f"index {idx[i]} out of range for dimension {i} with size {shape[i]}"
                    )
                flat_idx += dim_idx * stride
                stride *= shape[i]
            lib.tensor_set_float(self._tensor, flat_idx, float(value))
        else:
            raise TypeError(f"indices must be integers or tuples, not {type(idx).__name__}")

    @staticmethod
    def zeros(shape, dtype=None, device=None):
        return _create(lib.cml_zeros, shape, dtype, device)

    @staticmethod
    def ones(shape, dtype=None, device=None):
        return _create(lib.cml_ones, shape, dtype, device)

    @staticmethod
    def randn(shape, dtype=None, device=None):
        return _create(lib.cml_randn, shape, dtype, device)

    @staticmethod
    def rand(shape, dtype=None, device=None):
        return _create(lib.cml_rand, shape, dtype, device)

    @staticmethod
    def full(shape, value, dtype=None, device=None):
        return _create(lib.cml_full, shape, dtype, device, float(value))

    @staticmethod
    def empty(shape, dtype=None, device=None):
        return _create(lib.cml_empty, shape, dtype, device)

    @staticmethod
    def logspace(start: float, end: float, steps: int = 50) -> "Tensor":
        return Tensor.from_numpy(np.logspace(start, end, steps, dtype=np.float32))

    @staticmethod
    def stack(tensors: list, dim: int = 0) -> "Tensor":
        if not tensors:
            raise ValueError("Need at least one tensor to stack")
        c_tensors = ffi.new("Tensor*[]", [t._tensor for t in tensors])
        return Tensor(lib.cml_stack(c_tensors, len(tensors), dim))

    @staticmethod
    def cat(tensors: list, dim: int = 0) -> "Tensor":
        if not tensors:
            raise ValueError("Need at least one tensor")
        c_tensors = ffi.new("Tensor*[]", [t._tensor for t in tensors])
        return Tensor(lib.cml_concat(c_tensors, len(tensors), dim))

    @staticmethod
    def eye(n: int, m: Optional[int] = None, dtype=None, device=None) -> "Tensor":
        config = _make_config(dtype, device)
        return Tensor(lib.cml_eye(n, config))

    @staticmethod
    def arange(start: float, end: float = None, step: float = 1.0, dtype=None, device=None) -> "Tensor":
        if end is None:
            end = float(start)
            start = 0.0
        config = _make_config(dtype, device)
        return Tensor(lib.cml_arange(float(start), float(end), float(step), config))

    @staticmethod
    def linspace(start: float, end: float, steps: int = 100, dtype=None, device=None) -> "Tensor":
        config = _make_config(dtype, device)
        return Tensor(lib.cml_linspace(float(start), float(end), int(steps), config))


class _TensorView(Tensor):
    """Non-owning view; does not free the underlying C tensor."""

    def __init__(self, c_tensor):
        self._tensor = c_tensor
        self._shape_cache = None

    def __del__(self):
        pass


class init_context:
    def __init__(self):
        self._initialized = False

    def __enter__(self):
        lib.cml_init()
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._initialized:
            lib.cml_cleanup()
        return False


class set_grad_enabled:
    def __init__(self, mode: bool):
        self._mode = mode
        self._prev = True

    def __enter__(self):
        self._prev = lib.cml_is_grad_enabled()
        if self._mode:
            lib.cml_enable_grad()
        else:
            lib.cml_no_grad()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev:
            lib.cml_enable_grad()
        else:
            lib.cml_no_grad()
        return False


class no_grad(set_grad_enabled):
    def __init__(self):
        super().__init__(False)


class enable_grad(set_grad_enabled):
    def __init__(self):
        super().__init__(True)


def is_grad_enabled() -> bool:
    return lib.cml_is_grad_enabled()


def is_device_available(device: int) -> bool:
    if device == DEVICE_CPU:
        return True
    try:
        if device == DEVICE_CUDA:
            return bool(lib.device_cuda_available())
        elif device == DEVICE_METAL:
            return bool(lib.device_metal_available())
        elif device == DEVICE_ROCM:
            return bool(lib.device_rocm_available())
    except (AttributeError, Exception):
        pass
    return False
