#include <math.h>
#include <stdio.h>
#include "../../include/Optimizers/sgd.h"
#include "../../include/Optimizers/regularization.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

/**
 * @brief Performs the Stochastic Gradient Descent (SGD) optimization algorithm.
 *
 * SGD is a simple optimization algorithm that updates parameters using the gradient of the loss function.
 *
 * @param x The input feature value.
 * @param y The target value.
 * @param lr The learning rate.
 * @param w Pointer to the weight parameter.
 * @param b Pointer to the bias parameter.
 * @return The computed loss value, or an error code.
 */
float sgd(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b,
          SGDConfig config, int epoch)
{
    if (!w || !b || !v_w || !v_b)
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
    float dw = w_node->grad * (config.maximize ? -1.0f : 1.0f);
    float db = b_node->grad * (config.maximize ? -1.0f : 1.0f);

    // Update velocities and parameters using momentum if enabled
    if (config.momentum > 0.0f)
    {
        *v_w = config.momentum * (*v_w) + (1 - config.dampening) * dw;
        *v_b = config.momentum * (*v_b) + (1 - config.dampening) * db;

        if (config.nesterov)
        {
            *w -= lr * (dw + config.momentum * (*v_w));
            *b -= lr * (db + config.momentum * (*v_b));
        }
        else
        {
            *w -= lr * (*v_w);
            *b -= lr * (*v_b);
        }
    }
    else
    {
        *w -= lr * dw;
        *b -= lr * db;
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
