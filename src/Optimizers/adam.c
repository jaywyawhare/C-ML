#include <math.h>
#include <stdio.h>
#include "../../include/Optimizers/adam.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

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
float adam(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b, float *s_w, float *s_b, float beta1, float beta2, float epsilon)
{

    if (!w || !b || !v_w || !v_b || !s_w || !s_b)
    {
        fprintf(stderr, "[adam] Error: Null pointer input.\n");
        return CM_NULL_POINTER_ERROR;
    }

    if (epsilon <= 0 || beta1 >= 1.0 || beta2 >= 1.0 || beta1 <= 0.0 || beta2 <= 0.0 || lr <= 0 || isnan(x) || isnan(y) || isinf(x) || isinf(y))
    {
        fprintf(stderr, "[adam] Error: Invalid parameter(s) provided.\n");
        return CM_INVALID_INPUT_ERROR;
    }

    static int t = 0;
    t++;

    float y_pred = *w * x + *b;
    float loss = pow(y_pred - y, 2);

    float dw = 2 * (y_pred - y) * x;
    float db = 2 * (y_pred - y);

    *v_w = beta1 * *v_w + (1 - beta1) * dw;
    *v_b = beta1 * *v_b + (1 - beta1) * db;
    *s_w = beta2 * *s_w + (1 - beta2) * dw * dw;
    *s_b = beta2 * *s_b + (1 - beta2) * db * db;

    float v_w_corrected = *v_w / (1 - pow(beta1, t));
    float v_b_corrected = *v_b / (1 - pow(beta1, t));
    float s_w_corrected = *s_w / (1 - pow(beta2, t));
    float s_b_corrected = *s_b / (1 - pow(beta2, t));

    *w -= lr * v_w_corrected / (sqrt(s_w_corrected + epsilon));
    *b -= lr * v_b_corrected / (sqrt(s_b_corrected + epsilon));

#if DEBUG_LOGGING
    printf("[adam] w: %f, b: %f, loss: %f\n", *w, *b, loss);
#endif

    return loss;
}
