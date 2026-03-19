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
    """Cross entropy loss. Applies softmax internally."""
    result = lib.cml_nn_cross_entropy_loss(logits._tensor, labels._tensor)
    return Tensor(result)


def bce_loss(predictions, targets):
    """Binary cross entropy loss. Expects pre-sigmoid predictions."""
    result = lib.cml_nn_bce_loss(predictions._tensor, targets._tensor)
    return Tensor(result)


def huber_loss(predictions, targets, delta=1.0):
    result = lib.cml_nn_huber_loss(predictions._tensor, targets._tensor, float(delta))
    return Tensor(result)


def kl_div_loss(input, target):
    result = lib.cml_nn_kl_div_loss(input._tensor, target._tensor)
    return Tensor(result)


# Keep old name as alias for backwards compatibility
kl_divergence = kl_div_loss


def nll_loss(log_probs, targets):
    result = lib.cml_nn_nll_loss(log_probs._tensor, targets._tensor)
    return Tensor(result)


def sparse_cross_entropy_loss(input, target):
    result = lib.cml_nn_sparse_cross_entropy_loss(input._tensor, target._tensor)
    return Tensor(result)


def triplet_margin_loss(anchor, positive, negative, margin=1.0):
    result = lib.cml_nn_triplet_margin_loss(
        anchor._tensor, positive._tensor, negative._tensor, float(margin)
    )
    return Tensor(result)


def cosine_embedding_loss(x1, x2, target, margin=0.0):
    result = lib.cml_nn_cosine_embedding_loss(
        x1._tensor, x2._tensor, target._tensor, float(margin)
    )
    return Tensor(result)
