"""
Loss functions for training.
"""

from cml._cml_lib import ffi, lib
from cml.core import Tensor


def mse_loss(predictions, targets):
    """Mean Squared Error loss.

    MSE = mean((predictions - targets)^2)

    Good for regression tasks.

    Args:
        predictions: Predicted values
        targets: Target values

    Returns:
        Loss tensor

    Example:
        >>> output = model(x)
        >>> loss = mse_loss(output, y)
        >>> backward(loss)
    """
    result = lib.cml_nn_mse_loss(predictions._tensor, targets._tensor)
    return Tensor(result)


def mae_loss(predictions, targets):
    """Mean Absolute Error loss.

    MAE = mean(|predictions - targets|)

    More robust to outliers than MSE.

    Args:
        predictions: Predicted values
        targets: Target values

    Returns:
        Loss tensor

    Example:
        >>> loss = mae_loss(output, y)
    """
    result = lib.cml_nn_mae_loss(predictions._tensor, targets._tensor)
    return Tensor(result)


def cross_entropy_loss(logits, labels):
    """Cross Entropy loss with softmax.

    For multi-class classification. Applies softmax internally.

    Args:
        logits: Raw network outputs (shape: [batch, num_classes])
        labels: Class indices (shape: [batch])

    Returns:
        Loss tensor

    Example:
        >>> logits = model(x)
        >>> loss = cross_entropy_loss(logits, labels)
        >>> backward(loss)
    """
    result = lib.cml_nn_cross_entropy_loss(logits._tensor, labels._tensor)
    return Tensor(result)


def bce_loss(predictions, targets):
    """Binary Cross Entropy loss.

    For binary classification tasks.

    Args:
        predictions: Predicted probabilities (should be in [0, 1])
        targets: Binary targets (0 or 1)

    Returns:
        Loss tensor

    Note:
        Predictions should pass through sigmoid before this loss.

    Example:
        >>> probs = sigmoid(model(x))
        >>> loss = bce_loss(probs, y)
    """
    result = lib.cml_nn_bce_loss(predictions._tensor, targets._tensor)
    return Tensor(result)


def huber_loss(predictions, targets, delta=1.0):
    """Huber loss.

    Combines MSE and MAE. Less sensitive to outliers than MSE.

    Args:
        predictions: Predicted values
        targets: Target values
        delta: Transition point between L2 and L1 loss

    Returns:
        Loss tensor

    Example:
        >>> loss = huber_loss(output, y, delta=1.0)
    """
    result = lib.cml_nn_huber_loss(predictions._tensor, targets._tensor, delta)
    return Tensor(result)


def kl_divergence(p_logits, q_logits):
    """Kullback-Leibler divergence.

    Measures how one probability distribution differs from another.

    Args:
        p_logits: Reference distribution logits
        q_logits: Target distribution logits

    Returns:
        Loss tensor

    Example:
        >>> kl_loss = kl_divergence(teacher_logits, student_logits)
    """
    result = lib.cml_nn_kl_divergence(p_logits._tensor, q_logits._tensor)
    return Tensor(result)
