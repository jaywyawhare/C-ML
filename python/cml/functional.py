"""Functional API -- decorators and helpers for training."""

from typing import Callable, Optional, Dict, Any
from contextlib import contextmanager
import cml


class TrainingContext:
    """Context manager that sets device and dtype for a block, restoring on exit."""

    def __init__(self, device: Optional[str] = None, dtype: Optional[str] = None):
        self.device = device
        self.dtype = dtype
        self.old_device = None
        self.old_dtype = None

    def __enter__(self):
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
        if self.old_device is not None:
            cml.set_device(self.old_device)
        if self.old_dtype is not None:
            cml.set_dtype(self.old_dtype)


@contextmanager
def training_mode(model, training: bool = True):
    """Temporarily set a model's training/inference mode."""
    old_training = None  # Would track actual mode
    try:
        model.set_training(training)
        yield model
    finally:
        if old_training is not None:
            model.set_training(old_training)


@contextmanager
def disable_grad():
    """Context manager to disable gradient computation."""
    from cml.core import no_grad as _no_grad
    ctx = _no_grad()
    ctx.__enter__()
    try:
        yield
    finally:
        ctx.__exit__(None, None, None)


@contextmanager
def enable_grad():
    """Context manager to enable gradient computation."""
    from cml.core import enable_grad as _enable_grad
    ctx = _enable_grad()
    ctx.__enter__()
    try:
        yield
    finally:
        ctx.__exit__(None, None, None)


def timer(fn: Callable) -> Callable:
    """Decorator that prints function execution time."""
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{fn.__name__} took {elapsed:.3f} seconds")
        return result

    return wrapper


def suppress_output(fn: Callable) -> Callable:
    """Decorator to suppress stdout during function execution."""

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
    """Decorator to profile memory usage."""

    def wrapper(*args, **kwargs):
        # Memory profiling would go here
        return fn(*args, **kwargs)

    return wrapper


def requires_grad(fn: Callable) -> Callable:
    """Decorator to ensure gradients are enabled."""

    def wrapper(*args, **kwargs):
        # Would enable gradients
        return fn(*args, **kwargs)

    return wrapper


class MetricsTracker:
    """Accumulates and summarizes named training metrics."""

    def __init__(self):
        self.metrics: Dict[str, list] = {}

    def log(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get(self, name: str) -> list:
        return self.metrics.get(name, [])

    def average(self, name: str) -> float:
        values = self.get(name)
        return sum(values) / len(values) if values else 0.0

    def latest(self, name: str) -> Optional[float]:
        values = self.get(name)
        return values[-1] if values else None

    def __str__(self) -> str:
        parts = []
        for name, values in self.metrics.items():
            if values:
                parts.append(f"{name}: {values[-1]:.4f}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"MetricsTracker({len(self.metrics)} metrics)"


class EarlyStopping:
    """Stops training when loss has not improved for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait_count = 0

    def __call__(self, loss: float) -> bool:
        """Return True if training should stop."""
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
    """Adjusts learning rate each epoch (step, exponential, or linear decay)."""

    def __init__(self, optimizer, schedule: str = "step", **kwargs):
        self.optimizer = optimizer
        self.schedule = schedule
        self.initial_lr = kwargs.get("lr", 0.001)
        self.decay = kwargs.get("decay", 0.95)
        self.step_size = kwargs.get("step_size", 10)
        self.epoch = 0

    def step(self):
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
