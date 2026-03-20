#ifndef CML_AUTOGRAD_LOSS_FUNCTIONS_H
#define CML_AUTOGRAD_LOSS_FUNCTIONS_H

#include "autograd.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Tensor;

/* loss = mean((input - target)^2) */
struct Tensor* tensor_mse_loss(struct Tensor* input, struct Tensor* target);

/* loss = mean(|input - target|) */
struct Tensor* tensor_mae_loss(struct Tensor* input, struct Tensor* target);

/* loss = -mean(target * log(input) + (1 - target) * log(1 - input)) */
struct Tensor* tensor_bce_loss(struct Tensor* input, struct Tensor* target);

/* loss = -mean(target * log(softmax(input))) */
struct Tensor* tensor_cross_entropy_loss(struct Tensor* input, struct Tensor* target);

/*
 * Huber loss:
 *   |x| < delta: 0.5 * x^2
 *   otherwise:    delta * |x| - 0.5 * delta^2
 */
struct Tensor* tensor_huber_loss(struct Tensor* input, struct Tensor* target, float delta);

/* KL(P||Q) = sum(P * log(P / Q)), input=Q, target=P */
struct Tensor* tensor_kl_div_loss(struct Tensor* input, struct Tensor* target);

/* loss = mean(max(0, 1 - target * input)), target should be +1/-1 */
struct Tensor* tensor_hinge_loss(struct Tensor* input, struct Tensor* target);

/* Focal loss: -mean(alpha * (1 - p_t)^gamma * log(p_t)) */
struct Tensor* tensor_focal_loss(struct Tensor* input, struct Tensor* target, float alpha,
                                 float gamma);

/*
 * Smooth L1 loss:
 *   |x| < beta: 0.5 * x^2 / beta
 *   otherwise:   |x| - 0.5 * beta
 */
struct Tensor* tensor_smooth_l1_loss(struct Tensor* input, struct Tensor* target, float beta);

/* Numerically stable cross entropy from logits via log-sum-exp */
struct Tensor* tensor_sparse_cross_entropy_loss(struct Tensor* input, struct Tensor* target);

/* loss = mean(max(||a-p|| - ||a-n|| + margin, 0)) */
struct Tensor* tensor_triplet_margin_loss(struct Tensor* anchor, struct Tensor* positive,
                                          struct Tensor* negative, float margin);

struct Tensor* tensor_cosine_embedding_loss(struct Tensor* x1, struct Tensor* x2,
                                            struct Tensor* target, float margin);

/* loss = -mean(log_probs[i, targets[i]]) */
struct Tensor* tensor_nll_loss(struct Tensor* log_probs, struct Tensor* targets);

/* Cross entropy with label smoothing: (1-e)*CE + e*uniform_loss */
struct Tensor* tensor_cross_entropy_loss_smooth(struct Tensor* input, struct Tensor* target,
                                                float label_smoothing);
struct Tensor* tensor_sparse_cross_entropy_loss_smooth(struct Tensor* input, struct Tensor* target,
                                                       float label_smoothing);

#ifdef __cplusplus
}
#endif

#endif // CML_AUTOGRAD_LOSS_FUNCTIONS_H
