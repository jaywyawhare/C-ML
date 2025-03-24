#include <math.h>
#include <stdio.h>
#include "../../include/Optimizers/sgd.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

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
        fprintf(stderr, "[sgd] Error: Null pointer input.\n");
        return CM_NULL_POINTER_ERROR;
    }

    if (isnan(x) || isnan(y))
    {
        return NAN;
    }

    if (isinf(x) || isinf(y))
    {
        fprintf(stderr, "[sgd] Error: Invalid input (inf).\n");
        return CM_INVALID_INPUT_ERROR;
    }

    float y_pred = (*w) * x + (*b);
    float loss = pow(y_pred - y, 2);
    float dw = 2 * (y_pred - y) * x;
    float db = 2 * (y_pred - y);

    (*w) -= lr * dw;
    (*b) -= lr * db;

#if DEBUG_LOGGING
    printf("[sgd] w: %f, b: %f, loss: %f\n", *w, *b, loss);
#endif

    return loss;
}
