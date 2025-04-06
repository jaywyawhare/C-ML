#include <math.h>
#include <stdio.h>
#include "../../include/Optimizers/rmsprop.h"
#include "../../include/Core/error_codes.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

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
 * @param epsilon A small constant to prevent division by zero.
 * @param beta The decay rate for the moving average of squared gradients.
 * @return The computed loss value, or an error code.
 */
float rms_prop(float x, float y, float lr, float *w, float *b, float *cache_w, float *cache_b, float epsilon, float beta)
{
    if (!w || !b || !cache_w || !cache_b)
    {
        fprintf(stderr, "[rms_prop] Error: Null pointer input.\n");
        return CM_NULL_POINTER_ERROR;
    }

    if (isnan(x) || isnan(y))
    {
        return NAN;
    }

    if (isinf(x) || isinf(y))
    {
        return CM_INVALID_INPUT_ERROR;
    }

    if (epsilon <= 0)
    {
        fprintf(stderr, "[rms_prop] Error: Epsilon value (%f) is invalid.\n", epsilon);
        return CM_INVALID_INPUT_ERROR;
    }

    float y_pred = (*w) * x + (*b);
    float loss = pow(y_pred - y, 2);
    float dw = 2 * (y_pred - y) * x;
    float db = 2 * (y_pred - y);

    *cache_w = beta * (*cache_w) + (1 - beta) * (dw * dw);
    *cache_b = beta * (*cache_b) + (1 - beta) * (db * db);

    *w -= lr * (dw / (sqrt(*cache_w) + epsilon));
    *b -= lr * (db / (sqrt(*cache_b) + epsilon));

#if DEBUG_LOGGING
    printf("[rms_prop] Debug: w: %f, b: %f, loss: %f\n", *w, *b, loss);
#endif

    return loss;
}
