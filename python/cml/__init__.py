"""
CML - C Machine Learning Library Python Bindings

A pure Python interface to the high-performance C-ML library.

Example:
    >>> import cml
    >>> cml.init()
    >>>
    >>> # Create tensor
    >>> x = cml.randn([10, 20])
    >>>
    >>> # Build neural network
    >>> model = cml.Sequential()
    >>> model.add(cml.Linear(20, 128))
    >>> model.add(cml.ReLU())
    >>> model.add(cml.Linear(128, 10))
    >>>
    >>> # Create optimizer
    >>> optimizer = cml.Adam(model, lr=0.001)
    >>>
    >>> # Training loop
    >>> for epoch in range(10):
    ...     optimizer.zero_grad()
    ...     output = model(x)
    ...     loss = cml.mse_loss(output, target)
    ...     cml.backward(loss)
    ...     optimizer.step()
    ...
    >>> cml.cleanup()
"""

__version__ = "0.0.4"
__author__ = "C-ML Contributors"

# Import CFFI library
try:
    from cml._cml_lib import ffi, lib
except ImportError as e:
    raise ImportError(
        "CML bindings not found. Please build them first:\n"
        "  cd python\n"
        "  python3 cml/build_cffi.py\n"
        f"Original error: {e}"
    )

# Core imports
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
    # Context managers
    init_context,
    no_grad,
    enable_grad,
    set_grad_enabled,
    # Constants
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

# Convenience APIs
from cml.simple_api import (
    build_model,
    train_model,
    evaluate_model,
    predict,
    create_optimizer,
    get_loss_function,
    batch_iterator,
)

# Data utilities
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

# Functional utilities
from cml.functional import (
    TrainingContext,
    training_mode,
    disable_grad,
    enable_grad,
    MetricsTracker,
    EarlyStopping,
    LearningRateScheduler,
)

__all__ = [
    # Core
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
    # Context managers
    "init_context",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    # Constants
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
    # Backward compat (also in constants)
    "DTYPE_FLOAT32",
    "DTYPE_FLOAT64",
    # Tensor creation
    "zeros",
    "ones",
    "randn",
    "rand",
    "full",
    "clone",
    # Autograd
    "backward",
    "get_grad",
    "enable_grad",
    "disable_grad",
    # Neural Networks
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
    # Loss functions
    "mse_loss",
    "mae_loss",
    "cross_entropy_loss",
    "bce_loss",
    "huber_loss",
    "kl_divergence",
    # Optimizers
    "Adam",
    "SGD",
    "RMSprop",
    "AdaGrad",
    # Utilities
    "set_log_level",
    "get_error",
    "clear_error",
    # Convenience APIs
    "build_model",
    "train_model",
    "evaluate_model",
    "predict",
    "create_optimizer",
    "get_loss_function",
    # Data utilities
    "Dataset",
    "DataLoader",
    "create_dataset",
    "create_dataloader",
    "train_test_split",
    "normalize",
    "minmax_scale",
    "one_hot_encode",
    # Functional utilities
    "TrainingContext",
    "training_mode",
    "disable_grad",
    "enable_grad",
    "MetricsTracker",
    "EarlyStopping",
    "LearningRateScheduler",
]
