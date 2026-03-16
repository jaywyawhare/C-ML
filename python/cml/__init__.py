"""Python bindings for the C-ML high-performance machine learning library."""

__version__ = "0.0.4"
__author__ = "C-ML Contributors"

try:
    from cml._cml_lib import ffi, lib
except ImportError as e:
    raise ImportError(
        "CML bindings not found. Please build them first:\n"
        "  cd python\n"
        "  python3 cml/build_cffi.py\n"
        f"Original error: {e}"
    )

from cml.core import (
    init,
    cleanup,
    seed,
    Tensor,
    get_device,
    set_device,
    set_dtype,
    get_dtype,
    is_device_available,
    is_grad_enabled,
    init_context,
    no_grad,
    enable_grad,
    set_grad_enabled,
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_METAL,
    DEVICE_ROCM,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_INT64,
    DEVICE_NAMES,
    DTYPE_NAMES,
    DTYPE_TO_NUMPY,
    NUMPY_TO_DTYPE,
)

from cml.tensor_ops import (
    zeros,
    ones,
    randn,
    rand,
    full,
    clone,
)

from cml.autograd import (
    backward,
    get_grad,
    enable_grad,
    disable_grad,
)

from cml.nn import (
    Sequential,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Dropout,
    BatchNorm2d,
    LayerNorm,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
)

from cml.losses import (
    mse_loss,
    mae_loss,
    cross_entropy_loss,
    bce_loss,
    huber_loss,
    kl_divergence,
)

from cml.optim import (
    Adam,
    SGD,
    RMSprop,
    AdaGrad,
)

from cml.utils import (
    set_log_level,
    get_error,
    clear_error,
)

from cml.simple_api import (
    build_model,
    train_model,
    evaluate_model,
    predict,
    create_optimizer,
    get_loss_function,
    batch_iterator,
)

from cml.data import (
    Dataset,
    DataLoader,
    create_dataset,
    create_dataloader,
    train_test_split,
    normalize,
    minmax_scale,
    one_hot_encode,
)

from cml.functional import (
    TrainingContext,
    training_mode,
    disable_grad,
    enable_grad,
    MetricsTracker,
    EarlyStopping,
    LearningRateScheduler,
)

from cml import distributed
from cml import zoo

__all__ = [
    "init",
    "cleanup",
    "seed",
    "Tensor",
    "get_device",
    "set_device",
    "set_dtype",
    "get_dtype",
    "is_device_available",
    "is_grad_enabled",
    "init_context",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "DEVICE_CPU",
    "DEVICE_CUDA",
    "DEVICE_METAL",
    "DEVICE_ROCM",
    "DTYPE_FLOAT32",
    "DTYPE_FLOAT64",
    "DTYPE_INT32",
    "DTYPE_INT64",
    "DEVICE_NAMES",
    "DTYPE_NAMES",
    "DTYPE_TO_NUMPY",
    "NUMPY_TO_DTYPE",
    "zeros",
    "ones",
    "randn",
    "rand",
    "full",
    "clone",
    "backward",
    "get_grad",
    "enable_grad",
    "disable_grad",
    "Sequential",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Dropout",
    "BatchNorm2d",
    "LayerNorm",
    "Conv2d",
    "MaxPool2d",
    "AvgPool2d",
    "mse_loss",
    "mae_loss",
    "cross_entropy_loss",
    "bce_loss",
    "huber_loss",
    "kl_divergence",
    "Adam",
    "SGD",
    "RMSprop",
    "AdaGrad",
    "set_log_level",
    "get_error",
    "clear_error",
    "build_model",
    "train_model",
    "evaluate_model",
    "predict",
    "create_optimizer",
    "get_loss_function",
    "Dataset",
    "DataLoader",
    "create_dataset",
    "create_dataloader",
    "train_test_split",
    "normalize",
    "minmax_scale",
    "one_hot_encode",
    "TrainingContext",
    "training_mode",
    "disable_grad",
    "enable_grad",
    "MetricsTracker",
    "EarlyStopping",
    "LearningRateScheduler",
]
