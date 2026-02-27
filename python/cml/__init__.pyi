"""Type stubs for CML Python bindings.

This provides type hints for IDE support and static type checking.
"""

from typing import (
    Union,
    Optional,
    Tuple,
    List,
    Sequence,
    Callable,
    Any,
    Iterator,
    Dict,
    overload,
)
import numpy as np
import numpy.typing as npt

__version__: str
__author__: str

# =============================================================================
# Type Aliases
# =============================================================================

Shape = Union[int, Tuple[int, ...], List[int]]
ArrayLike = Union["Tensor", np.ndarray, List[float], float]
Device = Union[int, str]
DType = Union[int, str]

# =============================================================================
# Constants
# =============================================================================

DEVICE_CPU: int
DEVICE_CUDA: int
DEVICE_METAL: int
DEVICE_ROCM: int

DTYPE_FLOAT32: int
DTYPE_FLOAT64: int
DTYPE_INT32: int
DTYPE_INT64: int

DEVICE_NAMES: Dict[int, str]
DTYPE_NAMES: Dict[int, str]
DTYPE_TO_NUMPY: Dict[int, type]
NUMPY_TO_DTYPE: Dict[type, int]

# =============================================================================
# Initialization
# =============================================================================

def init() -> None:
    """Initialize CML library."""
    ...

def cleanup() -> None:
    """Clean up CML resources."""
    ...

def seed(s: int) -> None:
    """Set random seed for reproducibility."""
    ...

# =============================================================================
# Device and DType Management
# =============================================================================

def get_device() -> int:
    """Get default device."""
    ...

def set_device(device: int) -> None:
    """Set default device."""
    ...

def get_dtype() -> int:
    """Get default data type."""
    ...

def set_dtype(dtype: int) -> None:
    """Set default data type."""
    ...

def is_device_available(device: int) -> bool:
    """Check if device is available."""
    ...

def is_grad_enabled() -> bool:
    """Check if gradient computation is enabled."""
    ...

# =============================================================================
# Context Managers
# =============================================================================

class init_context:
    """Context manager for automatic CML initialization and cleanup."""

    def __enter__(self) -> "init_context": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...

class no_grad:
    """Context manager to disable gradient computation."""

    def __enter__(self) -> "no_grad": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...

class enable_grad:
    """Context manager to enable gradient computation."""

    def __enter__(self) -> "enable_grad": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...

class set_grad_enabled:
    """Context manager to set gradient computation state."""

    def __init__(self, mode: bool) -> None: ...
    def __enter__(self) -> "set_grad_enabled": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...

# =============================================================================
# Tensor Class
# =============================================================================

class Tensor:
    """Python wrapper for CML tensor with NumPy integration."""

    # Properties
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def numel(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def dtype(self) -> int: ...
    @property
    def device(self) -> int: ...
    @property
    def requires_grad(self) -> bool: ...
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None: ...
    @property
    def grad(self) -> Optional["Tensor"]: ...
    @property
    def is_contiguous(self) -> bool: ...

    # Methods
    def requires_grad_(self, requires_grad: bool = True) -> "Tensor": ...
    def is_scalar(self) -> bool: ...
    def item(self) -> float: ...
    def numpy(self) -> np.ndarray: ...
    def __array__(self, dtype: Optional[type] = None) -> np.ndarray: ...
    @classmethod
    def from_numpy(cls, arr: np.ndarray, requires_grad: bool = False) -> "Tensor": ...
    def to(
        self, device: Optional[Device] = None, dtype: Optional[DType] = None
    ) -> "Tensor": ...
    def backward(self) -> None: ...
    def clone(self) -> "Tensor": ...
    def detach(self) -> "Tensor": ...
    def contiguous(self) -> "Tensor": ...
    def view(self, *new_shape: int) -> "Tensor": ...
    def reshape(self, *new_shape: int) -> "Tensor": ...
    def transpose(self, axis1: int = 0, axis2: int = 1) -> "Tensor": ...
    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor": ...
    def squeeze(self, dim: Optional[int] = None) -> "Tensor": ...
    def unsqueeze(self, dim: int) -> "Tensor": ...

    # Reduction operations
    def sum(self, dim: int = -1, keepdim: bool = False) -> "Tensor": ...
    def mean(self, dim: int = -1, keepdim: bool = False) -> "Tensor": ...

    # Activation functions
    def relu(self) -> "Tensor": ...
    def sigmoid(self) -> "Tensor": ...
    def tanh(self) -> "Tensor": ...
    def softmax(self, dim: int = 1) -> "Tensor": ...

    # Operators
    def __add__(self, other: "Tensor") -> "Tensor": ...
    def __sub__(self, other: "Tensor") -> "Tensor": ...
    def __mul__(self, other: "Tensor") -> "Tensor": ...
    def __rmul__(self, other: "Tensor") -> "Tensor": ...
    def __truediv__(self, other: "Tensor") -> "Tensor": ...
    def __matmul__(self, other: "Tensor") -> "Tensor": ...
    def __neg__(self) -> "Tensor": ...
    def __pos__(self) -> "Tensor": ...
    def __abs__(self) -> "Tensor": ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: Union[int, Tuple[int, ...]]) -> float: ...
    def __setitem__(self, idx: Union[int, Tuple[int, ...]], value: float) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

    # Static factory methods
    @staticmethod
    def zeros(shape: Shape) -> "Tensor": ...
    @staticmethod
    def ones(shape: Shape) -> "Tensor": ...
    @staticmethod
    def randn(shape: Shape) -> "Tensor": ...
    @staticmethod
    def rand(shape: Shape) -> "Tensor": ...
    @staticmethod
    def full(shape: Shape, value: float) -> "Tensor": ...
    @staticmethod
    def empty(shape: Shape) -> "Tensor": ...

# =============================================================================
# Tensor Creation Functions
# =============================================================================

def zeros(shape: Shape) -> Tensor: ...
def ones(shape: Shape) -> Tensor: ...
def randn(shape: Shape) -> Tensor: ...
def rand(shape: Shape) -> Tensor: ...
def full(shape: Shape, value: float) -> Tensor: ...
def clone(tensor: Tensor) -> Tensor: ...

# =============================================================================
# Autograd
# =============================================================================

def backward(
    tensor: Tensor,
    gradient: Optional[Tensor] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> None: ...
def get_grad(tensor: Tensor) -> Optional[Tensor]: ...

# =============================================================================
# Neural Network Modules
# =============================================================================

class Module:
    """Base class for neural network modules."""

    def __call__(self, input: Tensor) -> Tensor: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def set_training(self, mode: bool) -> None: ...
    def is_training(self) -> bool: ...
    def train(self) -> "Module": ...
    def eval(self) -> "Module": ...

class Sequential(Module):
    """Sequential container for layers."""

    def __init__(self) -> None: ...
    def add(self, layer: Module) -> "Sequential": ...
    def __call__(self, input: Tensor) -> Tensor: ...

class Linear(Module):
    """Linear (fully connected) layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: int = ...,
        device: int = ...,
        bias: bool = True,
    ) -> None: ...

class ReLU(Module):
    """ReLU activation."""

    def __init__(self, inplace: bool = False) -> None: ...

class Sigmoid(Module):
    """Sigmoid activation."""

    def __init__(self) -> None: ...

class Tanh(Module):
    """Tanh activation."""

    def __init__(self) -> None: ...

class Softmax(Module):
    """Softmax activation."""

    def __init__(self, dim: int = 1) -> None: ...

class Dropout(Module):
    """Dropout regularization."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...

class BatchNorm2d(Module):
    """Batch normalization for 2D inputs."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None: ...

class LayerNorm(Module):
    """Layer normalization."""

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None: ...

class Conv2d(Module):
    """2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ) -> None: ...

class MaxPool2d(Module):
    """2D max pooling."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None: ...

class AvgPool2d(Module):
    """2D average pooling."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None: ...

# =============================================================================
# Loss Functions
# =============================================================================

def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor: ...
def mae_loss(predictions: Tensor, targets: Tensor) -> Tensor: ...
def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor: ...
def bce_loss(predictions: Tensor, targets: Tensor) -> Tensor: ...
def huber_loss(predictions: Tensor, targets: Tensor, delta: float = 1.0) -> Tensor: ...
def kl_divergence(p: Tensor, q: Tensor) -> Tensor: ...

# =============================================================================
# Optimizers
# =============================================================================

class Optimizer:
    """Base class for optimizers."""

    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
    def set_lr(self, lr: float) -> None: ...

class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        model: Module,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None: ...

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(
        self,
        model: Module,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None: ...

class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        model: Module,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None: ...

class AdaGrad(Optimizer):
    """AdaGrad optimizer."""

    def __init__(
        self,
        model: Module,
        lr: float = 0.01,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None: ...

# =============================================================================
# Utilities
# =============================================================================

def set_log_level(level: int) -> None: ...
def get_error() -> Optional[str]: ...
def clear_error() -> None: ...

# =============================================================================
# Convenience APIs
# =============================================================================

def build_model(layers: List[Dict[str, Any]]) -> Sequential: ...
def train_model(
    model: Module,
    X: Tensor,
    y: Tensor,
    optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    epochs: int,
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict[str, List[float]]: ...
def evaluate_model(
    model: Module,
    X: Tensor,
    y: Tensor,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> Dict[str, float]: ...
def predict(model: Module, X: Tensor) -> Tensor: ...
def create_optimizer(
    name: str,
    model: Module,
    lr: float = 0.001,
    **kwargs: Any,
) -> Optimizer: ...
def get_loss_function(name: str) -> Callable[[Tensor, Tensor], Tensor]: ...
def batch_iterator(
    X: Tensor,
    y: Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Tuple[Tensor, Tensor]]: ...

# =============================================================================
# Data Utilities
# =============================================================================

class Dataset:
    """Base dataset class."""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]: ...

class DataLoader:
    """Data loader for batching."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None: ...
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]: ...
    def __len__(self) -> int: ...

def create_dataset(X: np.ndarray, y: np.ndarray) -> Dataset: ...
def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader: ...
def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
def normalize(X: np.ndarray, axis: int = 0) -> np.ndarray: ...
def minmax_scale(X: np.ndarray, axis: int = 0) -> np.ndarray: ...
def one_hot_encode(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray: ...

# =============================================================================
# Functional Utilities
# =============================================================================

class TrainingContext:
    """Context manager for training configuration."""

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None: ...
    def __enter__(self) -> "TrainingContext": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

def training_mode(model: Module, training: bool = True) -> Any: ...

class MetricsTracker:
    """Track training metrics."""

    def __init__(self) -> None: ...
    def log(self, name: str, value: float) -> None: ...
    def get(self, name: str) -> List[float]: ...
    def average(self, name: str) -> float: ...
    def latest(self, name: str) -> Optional[float]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None: ...
    def __call__(self, loss: float) -> bool: ...

class LearningRateScheduler:
    """Learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        schedule: str = "step",
        **kwargs: Any,
    ) -> None: ...
    def step(self) -> None: ...
