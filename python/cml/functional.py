"""Training decorators and helpers."""

from typing import Callable, Optional, Dict, Any
from contextlib import contextmanager
import cml


class TrainingContext:
    """Sets device/dtype for a block, restoring on exit."""

    def __init__(self, device: Optional[str] = None, dtype: Optional[str] = None):
        """Record the target ``device``/``dtype`` names to apply on entry."""
        self.device = device
        self.dtype = dtype
        self.old_device = None
        self.old_dtype = None

    def __enter__(self):
        """Set the requested global device/dtype, stashing the previous values for restore."""
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
        """Restore the device/dtype that were in effect before the block."""
        if self.old_device is not None:
            cml.set_device(self.old_device)
        if self.old_dtype is not None:
            cml.set_dtype(self.old_dtype)


@contextmanager
def training_mode(model, training: bool = True):
    """Context manager toggling ``model`` train/eval mode, restoring the prior state on exit."""
    old_training = model.is_training() if hasattr(model, 'is_training') else not training
    try:
        model.set_training(training)
        yield model
    finally:
        model.set_training(old_training)


@contextmanager
def disable_grad():
    """Context manager disabling autograd for the block; like torch.no_grad."""
    from cml.core import no_grad as _no_grad
    ctx = _no_grad()
    ctx.__enter__()
    try:
        yield
    finally:
        ctx.__exit__(None, None, None)


@contextmanager
def enable_grad():
    """Context manager enabling autograd for the block; like torch.enable_grad."""
    from cml.core import enable_grad as _enable_grad
    ctx = _enable_grad()
    ctx.__enter__()
    try:
        yield
    finally:
        ctx.__exit__(None, None, None)


def timer(fn: Callable) -> Callable:
    """Decorator printing the wall-clock time taken by ``fn`` on each call."""
    import time

    def wrapper(*args, **kwargs):
        """Time the wrapped call and print its elapsed duration."""
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{fn.__name__} took {elapsed:.3f} seconds")
        return result

    return wrapper


def suppress_output(fn: Callable) -> Callable:
    """Decorator silencing stdout produced while ``fn`` runs."""
    def wrapper(*args, **kwargs):
        """Redirect stdout to a throwaway buffer for the wrapped call."""
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


class MetricsTracker:
    def __init__(self):
        """Create an empty tracker mapping metric names to their logged values."""
        self.metrics: Dict[str, list] = {}

    def log(self, name: str, value: float):
        """Append a value to the named metric's history."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get(self, name: str) -> list:
        """Return the full history of the named metric, or an empty list."""
        return self.metrics.get(name, [])

    def average(self, name: str) -> float:
        """Return the mean of the named metric's values, or 0.0 if none."""
        values = self.get(name)
        return sum(values) / len(values) if values else 0.0

    def latest(self, name: str) -> Optional[float]:
        """Return the most recent value of the named metric, or None."""
        values = self.get(name)
        return values[-1] if values else None

    def __str__(self) -> str:
        """Return a one-line summary of each metric's latest value."""
        parts = []
        for name, values in self.metrics.items():
            if values:
                parts.append(f"{name}: {values[-1]:.4f}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        """Return a short representation naming the number of tracked metrics."""
        return f"MetricsTracker({len(self.metrics)} metrics)"


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Configure how long to wait, and by how much loss must improve, before stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait_count = 0

    def __call__(self, loss: float) -> bool:
        """Record a loss and return True once it has stalled for ``patience`` calls."""
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
    """Step and exponential decay, driven from Python.

    ``cml.optim`` wraps the C ``LRScheduler`` and covers far more policies
    (StepLR, ExponentialLR, CosineAnnealingLR, ReduceOnPlateau, OneCycleLR,
    MultiStepLR, PolynomialLR, WarmupLR); prefer those. This stays for the two
    simple cases and for optimizers driven entirely from Python.
    """

    _SCHEDULES = ("step", "exponential")

    def __init__(self, optimizer, schedule: str = "step", **kwargs):
        """Configure a step or exponential decay schedule over the given optimizer."""
        if schedule not in self._SCHEDULES:
            # Anything else used to fall through step() silently, leaving the
            # learning rate untouched for the whole run.
            raise ValueError(
                f"unknown schedule {schedule!r}; this class supports "
                f"{self._SCHEDULES}. For cosine/plateau/warmup and friends use "
                f"the C-backed schedulers in cml.optim."
            )
        self.optimizer = optimizer
        self.schedule = schedule
        self.initial_lr = kwargs.get("lr", 0.001)
        self.decay = kwargs.get("decay", 0.95)
        self.step_size = kwargs.get("step_size", 10)
        self.epoch = 0

    def step(self):
        """Advance one epoch and apply the scheduled learning rate to the optimizer."""
        if self.schedule == "step":
            if self.epoch % self.step_size == 0:
                new_lr = self.initial_lr * (
                    self.decay ** (self.epoch // self.step_size)
                )
                self.optimizer.set_lr(new_lr)
        elif self.schedule == "exponential":
            new_lr = self.initial_lr * (self.decay**self.epoch)
            self.optimizer.set_lr(new_lr)

        self.epoch += 1
