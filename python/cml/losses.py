"""Loss functions for training."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor


def mse_loss(predictions, targets):
    result = lib.cml_nn_mse_loss(predictions._tensor, targets._tensor)
    return Tensor(result)


def mae_loss(predictions, targets):
    result = lib.cml_nn_mae_loss(predictions._tensor, targets._tensor)
    return Tensor(result)


def cross_entropy_loss(logits, labels):
    """Applies softmax internally."""
    result = lib.cml_nn_cross_entropy_loss(logits._tensor, labels._tensor)
    return Tensor(result)


def bce_loss(predictions, targets):
    """Expects pre-sigmoid predictions."""
    result = lib.cml_nn_bce_loss(predictions._tensor, targets._tensor)
    return Tensor(result)


def huber_loss(predictions, targets, delta=1.0):
    result = lib.cml_nn_huber_loss(predictions._tensor, targets._tensor, delta)
    return Tensor(result)


def kl_divergence(p_logits, q_logits):
    result = lib.cml_nn_kl_divergence(p_logits._tensor, q_logits._tensor)
    return Tensor(result)
