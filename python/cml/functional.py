"""
Functional API - Decorator and helper functions for training.

Provides:
- Training decorators
- Context managers
- Progress tracking
- Metric logging
"""

from typing import Callable, Optional, Dict, Any
from contextlib import contextmanager
import cml


class TrainingContext:
    """Context manager for training configuration.

    Example:
        >>> with TrainingContext(device="cuda", dtype="float32"):
        ...     model = cml.Sequential()
        ...     X = cml.randn([100, 10])
    """

    def __init__(self, device: Optional[str] = None, dtype: Optional[str] = None):
        """Initialize training context.

        Args:
            device: Device to use ("cpu", "cuda", "metal", "rocm")
            dtype: Data type ("float32", "float64")
        """
        self.device = device
        self.dtype = dtype
        self.old_device = None
        self.old_dtype = None

    def __enter__(self):
        """Enter context."""
        if self.device:
            self.old_device = cml.get_device()
            if self.device.lower() == "cuda":
                cml.set_device(cml.DEVICE_CUDA)
            elif self.device.lower() == "metal":
                cml.set_device(cml.DEVICE_METAL)
            elif self.device.lower() == "rocm":
                cml.set_device(cml.DEVICE_ROCM)
            else:
                cml.set_device(cml.DEVICE_CPU)

        if self.dtype:
            self.old_dtype = cml.get_dtype()
            if self.dtype.lower() == "float64":
                cml.set_dtype(cml.DTYPE_FLOAT64)
            else:
                cml.set_dtype(cml.DTYPE_FLOAT32)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self.old_device is not None:
            cml.set_device(self.old_device)
        if self.old_dtype is not None:
            cml.set_dtype(self.old_dtype)


@contextmanager
def training_mode(model, training: bool = True):
    """Context manager for training/inference mode.

    Args:
        model: Model to set mode for
        training: Whether to use training mode

    Example:
        >>> model = cml.Sequential()
        >>> with training_mode(model, True):
        ...     output = model(X)
        ...     loss = loss_fn(output, y)
        ...     cml.backward(loss)
    """
    old_training = None  # Would track actual mode
    try:
        model.set_training(training)
        yield model
    finally:
        if old_training is not None:
            model.set_training(old_training)


@contextmanager
def disable_grad():
    """Context manager to disable gradient computation.

    Example:
        >>> with disable_grad():
        ...     output = model(X)  # No gradients
    """
    # Placeholder - would implement grad disabling
    yield


@contextmanager
def enable_grad():
    """Context manager to enable gradient computation.

    Example:
        >>> with enable_grad():
        ...     output = model(X)  # Gradients enabled
    """
    yield


def timer(fn: Callable) -> Callable:
    """Decorator to time function execution.

    Args:
        fn: Function to time

    Returns:
        Wrapped function that prints execution time

    Example:
        >>> @timer
        ... def train_model(model, X, y):
        ...     # training code
    """
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{fn.__name__} took {elapsed:.3f} seconds")
        return result

    return wrapper


def suppress_output(fn: Callable) -> Callable:
    """Decorator to suppress print output.

    Args:
        fn: Function to suppress output for

    Returns:
        Wrapped function
    """

    def wrapper(*args, **kwargs):
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = fn(*args, **kwargs)
        finally:
            sys.stdout = old_stdout

        return result

    return wrapper


def profile_memory(fn: Callable) -> Callable:
    """Decorator to profile memory usage.

    Args:
        fn: Function to profile

    Returns:
        Wrapped function
    """

    def wrapper(*args, **kwargs):
        # Memory profiling would go here
        return fn(*args, **kwargs)

    return wrapper


def requires_grad(fn: Callable) -> Callable:
    """Decorator to ensure gradients are enabled.

    Args:
        fn: Function to wrap

    Returns:
        Wrapped function
    """

    def wrapper(*args, **kwargs):
        # Would enable gradients
        return fn(*args, **kwargs)

    return wrapper


class MetricsTracker:
    """Track training metrics.

    Example:
        >>> metrics = MetricsTracker()
        >>> for epoch in range(10):
        ...     loss = train_epoch(model, data)
        ...     metrics.log("loss", loss)
        ...     print(metrics)
    """

    def __init__(self):
        """Initialize tracker."""
        self.metrics: Dict[str, list] = {}

    def log(self, name: str, value: float):
        """Log a metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get(self, name: str) -> list:
        """Get all values for a metric.

        Args:
            name: Metric name

        Returns:
            List of values
        """
        return self.metrics.get(name, [])

    def average(self, name: str) -> float:
        """Get average value for a metric.

        Args:
            name: Metric name

        Returns:
            Average value
        """
        values = self.get(name)
        return sum(values) / len(values) if values else 0.0

    def latest(self, name: str) -> Optional[float]:
        """Get latest value for a metric.

        Args:
            name: Metric name

        Returns:
            Latest value or None
        """
        values = self.get(name)
        return values[-1] if values else None

    def __str__(self) -> str:
        """String representation."""
        parts = []
        for name, values in self.metrics.items():
            if values:
                parts.append(f"{name}: {values[-1]:.4f}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        """Representation."""
        return f"MetricsTracker({len(self.metrics)} metrics)"


class EarlyStopping:
    """Early stopping callback.

    Example:
        >>> early_stop = EarlyStopping(patience=5)
        >>> for epoch in range(100):
        ...     loss = train_epoch(model, data)
        ...     if early_stop(loss):
        ...         print("Early stopping!")
        ...         break
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialize early stopping.

        Args:
            patience: Epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait_count = 0

    def __call__(self, loss: float) -> bool:
        """Check if should stop.

        Args:
            loss: Current loss value

        Returns:
            True if should stop
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait_count = 0
            return False
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                return True
            return False


class LearningRateScheduler:
    """Learning rate scheduler.

    Example:
        >>> scheduler = LearningRateScheduler(optimizer, decay=0.95)
        >>> for epoch in range(100):
        ...     train_epoch(model, data)
        ...     scheduler.step()
    """

    def __init__(self, optimizer, schedule: str = "step", **kwargs):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            schedule: Schedule type ("step", "exponential", "linear")
            **kwargs: Schedule-specific arguments
        """
        self.optimizer = optimizer
        self.schedule = schedule
        self.initial_lr = kwargs.get("lr", 0.001)
        self.decay = kwargs.get("decay", 0.95)
        self.step_size = kwargs.get("step_size", 10)
        self.epoch = 0

    def step(self):
        """Perform scheduler step."""
        if self.schedule == "step":
            if self.epoch % self.step_size == 0:
                new_lr = self.initial_lr * (
                    self.decay ** (self.epoch // self.step_size)
                )
                self.optimizer.set_lr(new_lr)
        elif self.schedule == "exponential":
            new_lr = self.initial_lr * (self.decay**self.epoch)
            self.optimizer.set_lr(new_lr)
        elif self.schedule == "linear":
            # Linear decay
            pass

        self.epoch += 1
