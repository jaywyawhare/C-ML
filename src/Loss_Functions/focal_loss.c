#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/focal_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

#define EPSILON 1e-8

/**
 * @brief Computes the Focal Loss.
 *
 * The Focal Loss is designed to address class imbalance by down-weighting
 * well-classified examples. It is defined as:
 * - loss = -1/n * Σ [y * (1 - yHat)^gamma * log(yHat) + (1 - y) * yHat^gamma * log(1 - yHat)]
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted probabilities.
 * @param n The number of elements in y and yHat.
 * @param gamma The focusing parameter to adjust the rate at which easy examples are down-weighted.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float focal_loss(float *y, float *yHat, int n, float gamma)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float loss = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float yHat_clamped = fmaxf(fminf(yHat[i], 1.0f - EPSILON), EPSILON);
        if (y[i] == 1)
        {
            loss += -powf(1 - yHat_clamped, gamma) * logf(yHat_clamped);
        }
        else
        {
            loss += -powf(yHat_clamped, gamma) * logf(1 - yHat_clamped);
        }
    }
    return loss / n;
}

/**
 * @brief Computes the derivative of the Focal Loss.
 *
 * The derivative is defined as:
 * - d(Focal Loss)/dyHat = -gamma * (1 - yHat)^(gamma - 1) * log(yHat) * y
 *                         + (1 - y) * gamma * yHat^(gamma - 1) * log(1 - yHat)
 *
 * @param y Ground truth value.
 * @param yHat Predicted value.
 * @param gamma The focusing parameter.
 * @return The derivative value.
 */
float focal_loss_derivative(float y, float yHat, float gamma)
{
    float pt = fmaxf(yHat, 1e-15);
    float one_minus_pt = fmaxf(1 - yHat, 1e-15);

    return -y * gamma * powf(one_minus_pt, gamma - 1) * logf(pt) + (1 - y) * gamma * powf(pt, gamma - 1) * logf(one_minus_pt);
}