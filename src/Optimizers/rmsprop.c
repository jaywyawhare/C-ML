#include <math.h>
#include <stdio.h>
#include "../../include/Optimizers/rmsprop.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

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
        LOG_ERROR("Null pointer input.");
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
        LOG_ERROR("Epsilon value (%f) is invalid.", epsilon);
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
    LOG_DEBUG("w: %f, b: %f, loss: %f", *w, *b, loss);
#endif

    return loss;
}

/**
 * @brief Update weights and biases using RMSProp optimizer.
 *
 * Updates the weights and biases of a neural network using the RMSProp optimization algorithm.
 *
 * @param w Pointer to the weight.
 * @param b Pointer to the bias.
 * @param cache_w Pointer to the weight cache.
 * @param cache_b Pointer to the bias cache.
 * @param gradient Gradient value.
 * @param input Input value.
 * @param learning_rate Learning rate.
 * @param beta Decay rate for RMSProp.
 * @param epsilon Small value to prevent division by zero.
 */
void update_rmsprop(float *w, float *b, float *cache_w, float *cache_b, float gradient, float input, float learning_rate, float beta, float epsilon)
{
    *cache_w = beta * (*cache_w) + (1 - beta) * pow(gradient * input, 2);
    *cache_b = beta * (*cache_b) + (1 - beta) * pow(gradient, 2);

    *w -= learning_rate * (gradient * input) / (sqrt(*cache_w) + epsilon);
    *b -= learning_rate * gradient / (sqrt(*cache_b) + epsilon);
}
