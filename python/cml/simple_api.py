"""
Simple, high-level API for common ML tasks.

This module provides convenience functions for common operations,
making CML more accessible to beginners.

Example:
    >>> from cml.simple_api import *
    >>>
    >>> # Simple training loop
    >>> model = build_model([10, 20, 20, 5])
    >>> train_model(model, X, y, epochs=10)
    >>>
    >>> # Make predictions
    >>> predictions = model(X)
"""

from typing import List, Tuple, Optional, Callable
import cml
from cml.core import Tensor, DTYPE_FLOAT32, DEVICE_CPU
from cml.nn import Sequential, Linear, ReLU, Dropout


def build_model(
    layer_sizes: List[int], dropout: float = 0.0, activation: str = "relu"
) -> Sequential:
    """Build a simple feedforward neural network.

    Creates a Sequential model with specified layer sizes.
    Automatically adds activation functions between layers.

    Args:
        layer_sizes: List of layer sizes [input, hidden..., output]
        dropout: Dropout probability (0 = no dropout)
        activation: Activation function ("relu", "sigmoid", "tanh")

    Returns:
        Sequential model ready for training

    Example:
        >>> # Build 3-layer network: 784 -> 128 -> 64 -> 10
        >>> model = build_model([784, 128, 64, 10])
        >>>
        >>> # With dropout
        >>> model = build_model([784, 256, 10], dropout=0.5)
    """
    if len(layer_sizes) < 2:
        raise ValueError("Need at least 2 layer sizes (input and output)")

    model = Sequential()

    # Add layers
    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]
        is_last = i == len(layer_sizes) - 2

        # Add linear layer
        model.add(Linear(in_size, out_size))

        # Add activation (except for last layer)
        if not is_last:
            if activation == "relu":
                model.add(ReLU())
            elif activation == "sigmoid":
                model.add(cml.Sigmoid())
            elif activation == "tanh":
                model.add(cml.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Add dropout if specified
            if dropout > 0:
                model.add(Dropout(dropout))

    return model


def train_model(
    model: Sequential,
    X_train: Tensor,
    y_train: Tensor,
    epochs: int = 10,
    batch_size: Optional[int] = None,
    learning_rate: float = 0.001,
    loss_fn: str = "mse",
    optimizer: str = "adam",
    verbose: bool = True,
) -> List[float]:
    """Simple training loop for a model.

    Handles initialization, training loop, and cleanup automatically.

    Args:
        model: Sequential model to train
        X_train: Training features
        y_train: Training targets
        epochs: Number of training epochs
        batch_size: Batch size (None = full batch)
        learning_rate: Learning rate
        loss_fn: Loss function ("mse", "mae", "cross_entropy", "bce")
        optimizer: Optimizer ("adam", "sgd", "rmsprop", "adagrad")
        verbose: Print progress

    Returns:
        List of loss values for each epoch

    Example:
        >>> model = build_model([784, 128, 10])
        >>> X = cml.randn([1000, 784])
        >>> y = cml.zeros([1000, 10])
        >>> losses = train_model(model, X, y, epochs=5, learning_rate=0.01)
        >>> print(f"Final loss: {losses[-1]:.4f}")
    """
    # Determine loss function
    if loss_fn == "mse":
        loss_function = cml.mse_loss
    elif loss_fn == "mae":
        loss_function = cml.mae_loss
    elif loss_fn == "cross_entropy":
        loss_function = cml.cross_entropy_loss
    elif loss_fn == "bce":
        loss_function = cml.bce_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    # Create optimizer
    if optimizer == "adam":
        opt = cml.Adam(model, lr=learning_rate)
    elif optimizer == "sgd":
        opt = cml.SGD(model, lr=learning_rate, momentum=0.9)
    elif optimizer == "rmsprop":
        opt = cml.RMSprop(model, lr=learning_rate)
    elif optimizer == "adagrad":
        opt = cml.AdaGrad(model, lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Set training mode
    model.set_training(True)

    # Training loop
    losses = []
    for epoch in range(epochs):
        opt.zero_grad()

        # Forward pass
        output = model(X_train)

        # Compute loss
        loss = loss_function(output, y_train)

        # Backward pass
        cml.backward(loss)

        # Update
        opt.step()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs}: Loss computed")

        losses.append(loss)

    if verbose:
        print("Training complete!")

    return losses


def evaluate_model(
    model: Sequential, X_test: Tensor, y_test: Tensor, loss_fn: str = "mse"
) -> float:
    """Evaluate model on test data.

    Args:
        model: Model to evaluate
        X_test: Test features
        y_test: Test targets
        loss_fn: Loss function to use

    Returns:
        Loss value on test data

    Example:
        >>> test_loss = evaluate_model(model, X_test, y_test)
        >>> print(f"Test loss: {test_loss:.4f}")
    """
    # Set inference mode
    model.set_training(False)

    # Determine loss function
    if loss_fn == "mse":
        loss_function = cml.mse_loss
    elif loss_fn == "mae":
        loss_function = cml.mae_loss
    elif loss_fn == "cross_entropy":
        loss_function = cml.cross_entropy_loss
    elif loss_fn == "bce":
        loss_function = cml.bce_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    # Forward pass
    output = model(X_test)

    # Compute loss
    loss = loss_function(output, y_test)

    return loss


def predict(model: Sequential, X: Tensor) -> Tensor:
    """Make predictions on data.

    Sets model to inference mode and returns predictions.

    Args:
        model: Model to use
        X: Input data

    Returns:
        Model predictions

    Example:
        >>> predictions = predict(model, X_test)
    """
    model.set_training(False)
    return model(X)


def create_optimizer(
    model: Sequential, optimizer: str = "adam", learning_rate: float = 0.001, **kwargs
) -> cml.optim.Optimizer:
    """Create an optimizer for a model.

    Convenience function for creating optimizers with sensible defaults.

    Args:
        model: Model to optimize
        optimizer: Optimizer type ("adam", "sgd", "rmsprop", "adagrad")
        learning_rate: Learning rate
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance

    Example:
        >>> opt = create_optimizer(model, "adam", learning_rate=0.01)
        >>> opt = create_optimizer(model, "sgd", learning_rate=0.1, momentum=0.9)
    """
    if optimizer == "adam":
        return cml.Adam(
            model,
            lr=learning_rate,
            weight_decay=kwargs.get("weight_decay", 0.0),
            beta1=kwargs.get("beta1", 0.9),
            beta2=kwargs.get("beta2", 0.999),
        )
    elif optimizer == "sgd":
        return cml.SGD(
            model,
            lr=learning_rate,
            momentum=kwargs.get("momentum", 0.0),
            weight_decay=kwargs.get("weight_decay", 0.0),
        )
    elif optimizer == "rmsprop":
        return cml.RMSprop(
            model,
            lr=learning_rate,
            alpha=kwargs.get("alpha", 0.99),
            weight_decay=kwargs.get("weight_decay", 0.0),
        )
    elif optimizer == "adagrad":
        return cml.AdaGrad(model, lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def get_loss_function(loss_fn: str) -> Callable:
    """Get a loss function by name.

    Args:
        loss_fn: Loss function name

    Returns:
        Loss function

    Example:
        >>> loss_fn = get_loss_function("cross_entropy")
        >>> loss = loss_fn(output, target)
    """
    if loss_fn == "mse":
        return cml.mse_loss
    elif loss_fn == "mae":
        return cml.mae_loss
    elif loss_fn == "cross_entropy":
        return cml.cross_entropy_loss
    elif loss_fn == "bce":
        return cml.bce_loss
    elif loss_fn == "huber":
        return cml.huber_loss
    elif loss_fn == "kl":
        return cml.kl_divergence
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


def batch_iterator(X: Tensor, y: Tensor, batch_size: int):
    """Create batches for training.

    Generator that yields (X_batch, y_batch) tuples.

    Args:
        X: Features
        y: Targets
        batch_size: Batch size

    Yields:
        (X_batch, y_batch) tuples

    Example:
        >>> for X_batch, y_batch in batch_iterator(X, y, batch_size=32):
        ...     output = model(X_batch)
        ...     loss = loss_fn(output, y_batch)
    """
    num_samples = X.size  # Simplified

    for i in range(0, num_samples, batch_size):
        if i + batch_size <= num_samples:
            X_batch = X.slice(i, i + batch_size)
            y_batch = y.slice(i, i + batch_size)
            yield X_batch, y_batch
