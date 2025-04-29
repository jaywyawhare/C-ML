#include <math.h>
#include <stdio.h>
#include "../../include/Core/autograd.h"
#include "../../include/Optimizers/rmsprop.h"
#include "../../include/Optimizers/regularization.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Performs the RMSProp optimization algorithm.
 *
 * RMSProp is an adaptive learning rate optimization algorithm that adjusts the learning rate for each parameter.
 *
 * @param x The input feature value.
 * @param y The target value.
 * @param lr The learning rate.
 * @param w Pointer to the weight parameter.
 * @param b Pointer to the bias parameter.
 * @param cache_w Pointer to the cache for the weight parameter.
 * @param cache_b Pointer to the cache for the bias parameter.
 * @param v_w Pointer to the velocity for the weight parameter.
 * @param v_b Pointer to the velocity for the bias parameter.
 * @param beta The decay rate for the moving average of squared gradients.
 * @param epsilon A small constant to prevent division by zero.
 * @param momentum The momentum factor.
 * @param reg_type The type of regularization to use (L1, L2, or none).
 * @param lambda The regularization strength.
 * @param l1_ratio The ratio of L1 regularization.
 * @return The computed loss value, or an error code.
 */
float rmsprop(float x, float y, float lr, float *w, float *b,
              float *cache_w, float *cache_b, float *v_w, float *v_b,
              float *avg_grad_w, float *avg_grad_b, RMSpropConfig config,
              int epoch)
{
    if (!w || !b || !cache_w || !cache_b)
    {
        LOG_ERROR("Null pointer input.");
        return CM_NULL_POINTER_ERROR;
    }

    // Create autograd nodes
    Node *x_node = tensor(x, 0);
    Node *y_node = tensor(y, 0);
    Node *w_node = tensor(*w, 1);
    Node *b_node = tensor(*b, 1);

    // Forward pass with autograd
    Node *pred = add(mul(w_node, x_node), b_node);
    Node *diff = sub(pred, y_node);
    Node *loss = pow(diff, tensor(2.0f, 0));

    // Add regularization using autograd
    if (config.regularizer.type != NO_REGULARIZATION)
    {
        Node *reg = compute_regularization_node(w_node, config.regularizer);
        loss = add(loss, reg);
    }

    // Backward pass - get gradients automatically
    acc_grad(loss, 1.0f);
    float dw = w_node->grad;
    float db = b_node->grad;

    // Update cache
    *cache_w = config.alpha * (*cache_w) + (1 - config.alpha) * dw * dw;
    *cache_b = config.alpha * (*cache_b) + (1 - config.alpha) * db * db;

    // Update parameters with centered or standard RMSprop
    if (config.centered)
    {
        *avg_grad_w = config.alpha * (*avg_grad_w) + (1 - config.alpha) * dw;
        *avg_grad_b = config.alpha * (*avg_grad_b) + (1 - config.alpha) * db;

        float denom_w = sqrtf(*cache_w - (*avg_grad_w) * (*avg_grad_w) + config.eps);
        float denom_b = sqrtf(*cache_b - (*avg_grad_b) * (*avg_grad_b) + config.eps);

        *w -= lr * dw / denom_w;
        *b -= lr * db / denom_b;
    }
    else
    {
        *w -= lr * dw / sqrtf(*cache_w + config.eps);
        *b -= lr * db / sqrtf(*cache_b + config.eps);
    }

    float final_loss = loss->value;

    // Cleanup
    cm_safe_free((void **)&x_node);
    cm_safe_free((void **)&y_node);
    cm_safe_free((void **)&w_node);
    cm_safe_free((void **)&b_node);
    cm_safe_free((void **)&pred);
    cm_safe_free((void **)&diff);
    cm_safe_free((void **)&loss);

    return final_loss;
}
