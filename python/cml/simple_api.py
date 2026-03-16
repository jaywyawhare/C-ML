"""High-level convenience API for common ML tasks."""

from typing import List, Optional, Callable
import cml
from cml.core import Tensor
from cml.nn import Sequential, Linear, ReLU, Dropout


def build_model(
    layer_sizes: List[int], dropout: float = 0.0, activation: str = "relu"
) -> Sequential:
    if len(layer_sizes) < 2:
        raise ValueError("Need at least 2 layer sizes (input and output)")

    model = Sequential()

    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]
        is_last = i == len(layer_sizes) - 2

        model.add(Linear(in_size, out_size))

        if not is_last:
            if activation == "relu":
                model.add(ReLU())
            elif activation == "sigmoid":
                model.add(cml.Sigmoid())
            elif activation == "tanh":
                model.add(cml.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

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
    loss_function = get_loss_function(loss_fn)
    opt = create_optimizer(model, optimizer, learning_rate)

    model.set_training(True)

    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        output = model(X_train)
        loss = loss_function(output, y_train)
        cml.backward(loss)
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
    model.set_training(False)
    loss_function = get_loss_function(loss_fn)
    output = model(X_test)
    return loss_function(output, y_test)


def predict(model: Sequential, X: Tensor) -> Tensor:
    model.set_training(False)
    return model(X)


def create_optimizer(
    model: Sequential, optimizer: str = "adam", learning_rate: float = 0.001, **kwargs
) -> cml.optim.Optimizer:
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
    loss_map = {
        "mse": cml.mse_loss,
        "mae": cml.mae_loss,
        "cross_entropy": cml.cross_entropy_loss,
        "bce": cml.bce_loss,
        "huber": cml.huber_loss,
        "kl": cml.kl_divergence,
    }
    if loss_fn not in loss_map:
        raise ValueError(f"Unknown loss function: {loss_fn}")
    return loss_map[loss_fn]


def batch_iterator(X: Tensor, y: Tensor, batch_size: int):
    num_samples = X.size

    for i in range(0, num_samples, batch_size):
        if i + batch_size <= num_samples:
            X_batch = X.slice(i, i + batch_size)
            y_batch = y.slice(i, i + batch_size)
            yield X_batch, y_batch
