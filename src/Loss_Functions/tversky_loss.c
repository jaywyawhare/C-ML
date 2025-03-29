#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/tversky_loss.h"
#include "../../include/Core/error_codes.h"

#define ALPHA 0.5f
#define BETA 0.5f

/**
 * @brief Computes the Tversky Loss.
 *
 * The Tversky Loss is a generalization of the Dice Loss and is defined as:
 * - loss = 1 - (TP / (TP + alpha * FP + beta * FN))
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float tversky_loss(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[tversky_loss] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float tp = 0.0f, fp = 0.0f, fn = 0.0f;
    for (int i = 0; i < n; i++)
    {
        tp += y[i] * yHat[i];
        fp += (1 - y[i]) * yHat[i];
        fn += y[i] * (1 - yHat[i]);
    }
    float denominator = tp + ALPHA * fp + BETA * fn;
    if (denominator == 0)
    {
        fprintf(stderr, "[tversky_loss] Error: Division by zero.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    return 1.0f - (tp / denominator);
}

/**
 * @brief Computes the derivative of the Tversky Loss.
 *
 * The derivative is defined as:
 * - d(loss)/dyHat = -TP / (TP + alpha * FP + beta * FN)^2
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The derivative value, or an error code if inputs are invalid.
 */
float tversky_loss_derivative(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[tversky_loss_derivative] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float tp = 0.0f, fp = 0.0f, fn = 0.0f;
    for (int i = 0; i < n; i++)
    {
        tp += y[i] * yHat[i];
        fp += (1 - y[i]) * yHat[i];
        fn += y[i] * (1 - yHat[i]);
    }
    float denominator = tp + ALPHA * fp + BETA * fn;
    if (denominator == 0)
    {
        fprintf(stderr, "[tversky_loss_derivative] Error: Division by zero.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    return -tp / (denominator * denominator);
}