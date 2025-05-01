#include <math.h>
#include <stdio.h>
#include "../../include/Optimizers/adam.h"
#include "../../include/Optimizers/regularization.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"
#include "../../include/Core/memory_management.h"

/**
 * @brief Performs the Adam optimization algorithm.
 *
 * Adam is an adaptive learning rate optimization algorithm designed for training deep neural networks.
 *
 * @param x The input feature value.
 * @param y The target value.
 * @param lr The learning rate.
 * @param w Pointer to the weight parameter.
 * @param b Pointer to the bias parameter.
 * @param v_w Pointer to the first moment vector for the weight.
 * @param v_b Pointer to the first moment vector for the bias.
 * @param s_w Pointer to the second moment vector for the weight.
 * @param s_b Pointer to the second moment vector for the bias.
 * @param beta1 The exponential decay rate for the first moment estimates.
 * @param beta2 The exponential decay rate for the second moment estimates.
 * @param epsilon A small constant to prevent division by zero.
 * @return The computed loss value, or an error code.
 */
float adam(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b,
           float *s_w, float *s_b, float *max_s_w, float *max_s_b, AdamConfig config,
           int epoch)
{
    if (!w || !b || !v_w || !v_b || !s_w || !s_b)
    {
        LOG_ERROR("Null pointer input.");
        return CM_NULL_POINTER_ERROR;
    }

    // Create autograd nodes
    Node *lr_node = tensor(lr, 0);
    Node *x_node = tensor(x, 0);
    Node *y_node = tensor(y, 0);
    Node *w_node = tensor(*w, 1);
    Node *b_node = tensor(*b, 1);

    // Forward pass using autograd operations
    Node *adjusted_lr = adjust_learning_rate_node(lr_node, epoch, config.lr_scheduler,
                                                  config.lr_gamma, config.lr_step_size);
    Node *pred = tensor_add(tensor_mul(w_node, x_node), b_node);
    Node *diff = tensor_sub(pred, y_node);
    Node *loss = tensor_pow(diff, tensor(2.0f, 0));

    // Add regularization
    if (config.regularizer.type != NO_REGULARIZATION)
    {
        Node *reg = compute_regularization_node(w_node, config.regularizer);
        loss = tensor_add(loss, reg);
    }

    // Backward pass
    acc_grad(loss, 1.0f);

    // Get gradients
    float dw = w_node->grad * (config.maximize ? -1.0f : 1.0f);
    float db = b_node->grad * (config.maximize ? -1.0f : 1.0f);

    // Update using Adam
    float bc1 = 1.0f - powf(config.beta1, epoch + 1);
    float bc2 = 1.0f - powf(config.beta2, epoch + 1);

    // Update moments
    *v_w = config.beta1 * (*v_w) + (1 - config.beta1) * dw;
    *v_b = config.beta1 * (*v_b) + (1 - config.beta1) * db;
    *s_w = config.beta2 * (*s_w) + (1 - config.beta2) * dw * dw;
    *s_b = config.beta2 * (*s_b) + (1 - config.beta2) * db * db;

    // Parameter updates
    if (config.amsgrad)
    {
        *max_s_w = fmaxf(*max_s_w, *s_w);
        *max_s_b = fmaxf(*max_s_b, *s_b);
        *w -= (*v_w / bc1) / (sqrtf(*max_s_w / bc2) + config.epsilon);
        *b -= (*v_b / bc1) / (sqrtf(*max_s_b / bc2) + config.epsilon);
    }
    else
    {
        *w -= (*v_w / bc1) / (sqrtf(*s_w / bc2) + config.epsilon);
        *b -= (*v_b / bc1) / (sqrtf(*s_b / bc2) + config.epsilon);
    }

    float final_loss = loss->value;

    // Cleanup
    cm_safe_free((void **)&lr_node);
    cm_safe_free((void **)&adjusted_lr);
    cm_safe_free((void **)&x_node);
    cm_safe_free((void **)&y_node);
    cm_safe_free((void **)&w_node);
    cm_safe_free((void **)&b_node);
    cm_safe_free((void **)&pred);
    cm_safe_free((void **)&diff);
    cm_safe_free((void **)&loss);

    return final_loss;
}
