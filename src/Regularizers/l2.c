#include <math.h>
#include <stdio.h>
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"



/**
 * @brief Applies L2 regularization to update weights and biases.
 *
 * The L2 regularization adds the squared value of weights to the loss function
 * and updates the weights and biases using gradients and optimization parameters.
 *
 * @param x Input feature value.
 * @param y Target value.
 * @param lr Learning rate.
 * @param w Pointer to the weight.
 * @param b Pointer to the bias.
 * @param v_w Pointer to the velocity of the weight.
 * @param v_b Pointer to the velocity of the bias.
 * @param s_w Pointer to the squared gradient of the weight.
 * @param s_b Pointer to the squared gradient of the bias.
 * @param beta1 Exponential decay rate for the first moment estimates.
 * @param beta2 Exponential decay rate for the second moment estimates.
 * @param epsilon Small constant to prevent division by zero.
 * @param reg_l2 Regularization strength for L2.
 * @return The computed loss or an error code.
 */
float l2(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b, float *s_w, float *s_b, float beta1, float beta2, float epsilon, float reg_l2)
{
    if (w == NULL || b == NULL || v_w == NULL || v_b == NULL || s_w == NULL || s_b == NULL)
    {
        LOG_ERROR("Null pointer argument.");
        return CM_NULL_POINTER_ERROR;
    }

    if (epsilon <= 0)
    {
        LOG_ERROR("Epsilon must be positive.");
        return CM_INVALID_PARAMETER_ERROR;
    }

    if (lr <= 0 || beta1 >= 1.0 || beta2 >= 1.0 || beta1 <= 0.0 || beta2 <= 0.0 || isnan(x) || isnan(y) || isinf(x) || isinf(y) || isnan(reg_l2) || isinf(reg_l2))
    {
        LOG_ERROR("Invalid parameter(s) provided.");
        return CM_INVALID_INPUT_ERROR;
    }

    float y_pred = (*w) * x + (*b);
    float loss = pow(y_pred - y, 2) + reg_l2 * pow(*w, 2);
    float dw = 2 * (y_pred - y) * x + 2 * reg_l2 * (*w);
    float db = 2 * (y_pred - y);
    *v_w = beta1 * (*v_w) + (1 - beta1) * dw;
    *v_b = beta1 * (*v_b) + (1 - beta1) * db;
    *s_w = beta2 * (*s_w) + (1 - beta2) * pow(dw, 2);
    *s_b = beta2 * (*s_b) + (1 - beta2) * pow(db, 2);
    *w -= lr * ((*v_w) / (sqrt(*s_w) + epsilon));
    *b -= lr * ((*v_b) / (sqrt(*s_b) + epsilon));
    LOG_DEBUG("w: %f, b: %f, loss: %f", *w, *b, loss);

    return loss;
}
