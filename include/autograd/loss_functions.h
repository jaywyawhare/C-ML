/**
 * @file loss_functions.h
 * @brief Loss functions with autograd support
 *
 * This header provides loss functions with automatic differentiation.
 * All loss functions support automatic differentiation and can be used in
 * computation graphs.
 *
 * Loss functions are organized here separately from other operations.
 */

#ifndef CML_AUTOGRAD_LOSS_FUNCTIONS_H
#define CML_AUTOGRAD_LOSS_FUNCTIONS_H

#include "autograd.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct Tensor;

/**
 * @brief Mean Squared Error (MSE) Loss
 *
 * Computes the mean squared error between input and target tensors.
 * Mean Squared Error loss function.
 *
 * Formula: loss = mean((input - target)^2)
 *
 * @param input Predicted values tensor
 * @param target Target values tensor
 * @return Loss tensor (scalar) with autograd support, or NULL on failure
 *
 * @example
 * ```c
 * Tensor *prediction = ...;
 * Tensor *target = ...;
 * Tensor *loss = tensor_mse_loss(prediction, target);
 * tensor_backward(loss, NULL, false, false);
 * ```
 */
struct Tensor* tensor_mse_loss(struct Tensor* input, struct Tensor* target);

/**
 * @brief Mean Absolute Error (MAE) Loss
 *
 * Computes the mean absolute error between input and target tensors.
 * Mean Absolute Error loss function.
 *
 * Formula: loss = mean(|input - target|)
 *
 * @param input Predicted values tensor
 * @param target Target values tensor
 * @return Loss tensor (scalar) with autograd support, or NULL on failure
 */
struct Tensor* tensor_mae_loss(struct Tensor* input, struct Tensor* target);

/**
 * @brief Binary Cross Entropy Loss
 *
 * Computes the binary cross entropy loss between input and target.
 * Binary Cross Entropy loss function.
 *
 * Formula: loss = -mean(target * log(input) + (1 - target) * log(1 - input))
 *
 * @param input Predicted probabilities (should be in [0, 1])
 * @param target Target binary labels (0 or 1)
 * @return Loss tensor (scalar) with autograd support, or NULL on failure
 */
struct Tensor* tensor_bce_loss(struct Tensor* input, struct Tensor* target);

/**
 * @brief Cross Entropy Loss
 *
 * Computes the cross entropy loss for multi-class classification.
 * Cross Entropy loss function.
 *
 * Formula: loss = -mean(target * log(softmax(input)))
 *
 * @param input Predicted logits (before softmax)
 * @param target Target class indices
 * @return Loss tensor (scalar) with autograd support, or NULL on failure
 */
struct Tensor* tensor_cross_entropy_loss(struct Tensor* input, struct Tensor* target);

/**
 * @brief Huber Loss
 *
 * Computes the Huber loss (smooth L1 loss) between input and target.
 * Huber loss function.
 *
 * Formula:
 *   If |x| < delta: loss = 0.5 * x^2
 *   Else: loss = delta * |x| - 0.5 * delta^2
 *   where x = input - target
 *
 * @param input Predicted values tensor
 * @param target Target values tensor
 * @param delta Threshold parameter (default: 1.0)
 * @return Loss tensor (scalar) with autograd support, or NULL on failure
 */
struct Tensor* tensor_huber_loss(struct Tensor* input, struct Tensor* target, float delta);

/**
 * @brief KL Divergence Loss
 *
 * Computes the Kullback-Leibler divergence between two probability distributions.
 * KL Divergence loss function.
 *
 * Formula: KL(P||Q) = sum(P * log(P / Q))
 *
 * @param input Predicted distribution Q
 * @param target Target distribution P
 * @return Loss tensor (scalar) with autograd support, or NULL on failure
 */
struct Tensor* tensor_kl_div_loss(struct Tensor* input, struct Tensor* target);

/**
 * @brief Hinge Loss
 *
 * Computes the hinge loss for binary classification with SVM-style margins.
 *
 * Formula: loss = mean(max(0, 1 - target * input))
 *
 * @param input Predicted values tensor
 * @param target Target labels tensor (values should be +1 or -1)
 * @return Loss tensor (scalar) with autograd support, or NULL on failure
 */
struct Tensor* tensor_hinge_loss(struct Tensor* input, struct Tensor* target);

/**
 * @brief Focal Loss
 *
 * Computes the focal loss for handling class imbalance in binary classification.
 *
 * Formula: loss = -mean(alpha * (1 - p_t)^gamma * log(p_t))
 * where p_t = target * sigmoid(input) + (1 - target) * (1 - sigmoid(input))
 *
 * @param input Predicted logits tensor (before sigmoid)
 * @param target Target binary labels tensor (0 or 1)
 * @param alpha Balancing factor (default: 0.25)
 * @param gamma Focusing parameter (default: 2.0)
 * @return Loss tensor (scalar) with autograd support, or NULL on failure
 */
struct Tensor* tensor_focal_loss(struct Tensor* input, struct Tensor* target, float alpha,
                                 float gamma);

/**
 * @brief Smooth L1 Loss
 *
 * Computes the smooth L1 loss between input and target tensors.
 *
 * Formula:
 *   If |x| < beta: loss = 0.5 * x^2 / beta
 *   Else: loss = |x| - 0.5 * beta
 *   where x = input - target
 *
 * @param input Predicted values tensor
 * @param target Target values tensor
 * @param beta Threshold parameter (default: 1.0)
 * @return Loss tensor (scalar) with autograd support, or NULL on failure
 */
struct Tensor* tensor_smooth_l1_loss(struct Tensor* input, struct Tensor* target, float beta);

#ifdef __cplusplus
}
#endif

#endif // CML_AUTOGRAD_LOSS_FUNCTIONS_H
