#include <math.h>
#include <stdio.h>
#include "../../include/Optimizers/sgd.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

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
float sgd(float x, float y, float lr, float *w, float *b)
{
    if (!w || !b)
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
        LOG_ERROR("Invalid input (inf).");
        return CM_INVALID_INPUT_ERROR;
    }

    float y_pred = (*w) * x + (*b);
    float loss = pow(y_pred - y, 2);
    float dw = 2 * (y_pred - y) * x;
    float db = 2 * (y_pred - y);

    (*w) -= lr * dw;
    (*b) -= lr * db;

#if DEBUG_LOGGING
    LOG_DEBUG("w: %f, b: %f, loss: %f", *w, *b, loss);
#endif

    return loss;
}

/**
 * @brief Update weights and biases using SGD optimizer.
 *
 * Updates the weights and biases of a neural network using the SGD optimization algorithm.
 *
 * @param w Pointer to the weight.
 * @param b Pointer to the bias.
 * @param gradient Gradient value.
 * @param input Input value.
 * @param learning_rate Learning rate.
 */
void update_sgd(float *w, float *b, float gradient, float input, float learning_rate)
{
    *w -= learning_rate * gradient * input;
    *b -= learning_rate * gradient;
}
