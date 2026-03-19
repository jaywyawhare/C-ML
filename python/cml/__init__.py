"""Python bindings for the C-ML library."""

__version__ = "0.0.5"

try:
    from cml._cml_lib import ffi, lib
except ImportError as e:
    raise ImportError(
        "CML bindings not found. Build them first:\n"
        "  cd python && python3 cml/build_cffi.py\n"
        f"Original: {e}"
    )

from cml.core import (
    init, cleanup, seed,
    Tensor,
    get_device, set_device, get_dtype, set_dtype,
    is_device_available, is_grad_enabled,
    init_context, no_grad, enable_grad, set_grad_enabled,
    DEVICE_CPU, DEVICE_CUDA, DEVICE_METAL, DEVICE_ROCM,
    DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT32, DTYPE_INT64,
)

from cml.tensor_ops import (
    zeros, ones, rand, randn, full, empty, clone,
    arange, linspace, eye, randint, randperm, manual_seed,
    zeros_like, ones_like, rand_like, randn_like, full_like,
)

from cml.autograd import backward, get_grad, zero_grad

from cml import nn
from cml import optim
from cml import losses
from cml import distributed

from cml.losses import mse_loss, bce_loss, cross_entropy_loss
