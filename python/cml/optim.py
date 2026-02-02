"""
Optimization algorithms for training.
"""

from cml._cml_lib import ffi, lib


class Optimizer:
    """Base class for optimizers."""

    def __init__(self, c_optimizer):
        """Initialize optimizer.

        Args:
            c_optimizer: C optimizer pointer
        """
        self._optimizer = c_optimizer

    def step(self):
        """Perform optimization step (update parameters)."""
        lib.cml_optim_step(self._optimizer)

    def zero_grad(self):
        """Clear all parameter gradients."""
        lib.cml_optim_zero_grad(self._optimizer)

    def set_lr(self, lr):
        """Set learning rate.

        Args:
            lr: New learning rate value

        Example:
            >>> optimizer.set_lr(0.00001)  # Learning rate decay
        """
        lib.cml_optim_set_lr(self._optimizer, lr)

    def __del__(self):
        """Clean up optimizer."""
        if self._optimizer != ffi.NULL:
            lib.optimizer_free(self._optimizer)


class Adam(Optimizer):
    """Adam optimizer.

    Adaptive Moment Estimation. Combines momentum with adaptive learning rates.

    Good default choice for most problems.

    Example:
        >>> optimizer = Adam(model, lr=0.001)
        >>> for epoch in range(num_epochs):
        ...     optimizer.zero_grad()
        ...     output = model(x)
        ...     loss = loss_fn(output, y)
        ...     backward(loss)
        ...     optimizer.step()
    """

    def __init__(
        self, model, lr=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        """Initialize Adam optimizer.

        Args:
            model: Neural network module
            lr: Learning rate (default: 0.001)
            weight_decay: L2 regularization coefficient (default: 0.0)
            beta1: Exponential decay rate for first moment (default: 0.9)
            beta2: Exponential decay rate for second moment (default: 0.999)
            epsilon: Small constant for numerical stability (default: 1e-8)
        """
        optimizer = lib.cml_optim_adam_for_model(
            model._module, lr, weight_decay, beta1, beta2, epsilon
        )
        super().__init__(optimizer)


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    With momentum support for accelerated convergence.

    Example:
        >>> optimizer = SGD(model, lr=0.01, momentum=0.9)
    """

    def __init__(self, model, lr=0.01, momentum=0.0, weight_decay=0.0):
        """Initialize SGD optimizer.

        Args:
            model: Neural network module
            lr: Learning rate (default: 0.01)
            momentum: Momentum coefficient (default: 0.0)
            weight_decay: L2 regularization coefficient (default: 0.0)
        """
        optimizer = lib.cml_optim_sgd_for_model(
            model._module, lr, momentum, weight_decay
        )
        super().__init__(optimizer)


class RMSprop(Optimizer):
    """RMSprop optimizer.

    Divides learning rate by exponential moving average of gradient magnitudes.

    Example:
        >>> optimizer = RMSprop(model, lr=0.001)
    """

    def __init__(self, model, lr=0.001, alpha=0.99, epsilon=1e-8, weight_decay=0.0):
        """Initialize RMSprop optimizer.

        Args:
            model: Neural network module
            lr: Learning rate (default: 0.001)
            alpha: Decay rate (default: 0.99)
            epsilon: Small constant for numerical stability (default: 1e-8)
            weight_decay: L2 regularization coefficient (default: 0.0)
        """
        optimizer = lib.cml_optim_rmsprop_for_model(
            model._module, lr, alpha, epsilon, weight_decay
        )
        super().__init__(optimizer)


class AdaGrad(Optimizer):
    """AdaGrad optimizer.

    Adapts learning rate based on historical gradient information.

    Good for sparse data.

    Example:
        >>> optimizer = AdaGrad(model, lr=0.01)
    """

    def __init__(self, model, lr=0.01, epsilon=1e-10):
        """Initialize AdaGrad optimizer.

        Args:
            model: Neural network module
            lr: Learning rate (default: 0.01)
            epsilon: Small constant for numerical stability (default: 1e-10)
        """
        optimizer = lib.cml_optim_adagrad_for_model(model._module, lr, epsilon)
        super().__init__(optimizer)
