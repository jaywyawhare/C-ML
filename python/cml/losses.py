"""Loss functions for training."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor


def _wrap(result, name):
    """Wrap a C loss result, raising a clear error instead of returning a broken
    NULL tensor (the C loss returns NULL on invalid inputs, e.g. a wrong target
    shape)."""
    if result == ffi.NULL:
        raise RuntimeError(
            f"{name}: the C library returned NULL — check the argument shapes/"
            f"dtypes (e.g. cross_entropy_loss wants 1-D integer class-index "
            f"targets, not one-hot)."
        )
    return Tensor(result)


def mse_loss(predictions, targets):
    return _wrap(lib.cml_nn_mse_loss(predictions._tensor, targets._tensor), "mse_loss")


def mae_loss(predictions, targets):
    return _wrap(lib.cml_nn_mae_loss(predictions._tensor, targets._tensor), "mae_loss")


def cross_entropy_loss(logits, labels):
    """Cross entropy over logits (softmax applied internally).

    `labels` must be a 1-D tensor of integer class indices (PyTorch-style), one
    per sample — NOT one-hot vectors.
    """
    return _wrap(lib.cml_nn_cross_entropy_loss(logits._tensor, labels._tensor),
                 "cross_entropy_loss")


def bce_loss(predictions, targets):
    """Binary cross entropy loss. Expects pre-sigmoid predictions."""
    return _wrap(lib.cml_nn_bce_loss(predictions._tensor, targets._tensor), "bce_loss")


def huber_loss(predictions, targets, delta=1.0):
    return _wrap(lib.cml_nn_huber_loss(predictions._tensor, targets._tensor, float(delta)),
                 "huber_loss")


def kl_div_loss(input, target):
    return _wrap(lib.cml_nn_kl_div_loss(input._tensor, target._tensor), "kl_div_loss")


# Keep old name as alias for backwards compatibility
kl_divergence = kl_div_loss


def nll_loss(log_probs, targets):
    return _wrap(lib.cml_nn_nll_loss(log_probs._tensor, targets._tensor), "nll_loss")


def sparse_cross_entropy_loss(input, target):
    return _wrap(lib.cml_nn_sparse_cross_entropy_loss(input._tensor, target._tensor),
                 "sparse_cross_entropy_loss")


def triplet_margin_loss(anchor, positive, negative, margin=1.0):
    return _wrap(
        lib.cml_nn_triplet_margin_loss(anchor._tensor, positive._tensor,
                                       negative._tensor, float(margin)),
        "triplet_margin_loss")


def cosine_embedding_loss(x1, x2, target, margin=0.0):
    return _wrap(
        lib.cml_nn_cosine_embedding_loss(x1._tensor, x2._tensor, target._tensor,
                                         float(margin)),
        "cosine_embedding_loss")
