#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/cosine_similarity_loss.h"
#include "../../include/Core/error_codes.h"

/**
 * @brief Computes the Cosine Similarity Loss.
 *
 * The Cosine Similarity Loss is defined as:
 * - loss = 1 - (y . yHat) / (||y|| * ||yHat||)
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float cosine_similarity_loss(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[cosine_similarity_loss] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float dot_product = 0.0f, norm_y = 0.0f, norm_yHat = 0.0f;
    for (int i = 0; i < n; i++)
    {
        dot_product += y[i] * yHat[i];
        norm_y += y[i] * y[i];
        norm_yHat += yHat[i] * yHat[i];
    }
    norm_y = sqrtf(norm_y);
    norm_yHat = sqrtf(norm_yHat);
    if (norm_y == 0 || norm_yHat == 0)
    {
        fprintf(stderr, "[cosine_similarity_loss] Error: Zero vector norm.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }

    return 1.0f - (dot_product / (norm_y * norm_yHat));
}

/**
 * @brief Computes the derivative of the Cosine Similarity Loss.
 *
 * The derivative is defined as:
 * - d(loss)/dyHat = -y / (||y|| * ||yHat||) + (y . yHat) * yHat / (||yHat||^3)
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The derivative value, or an error code if inputs are invalid.
 */
float cosine_similarity_loss_derivative(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[cosine_similarity_loss_derivative] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float dot_product = 0.0f, norm_y = 0.0f, norm_yHat = 0.0f;
    for (int i = 0; i < n; i++)
    {
        dot_product += y[i] * yHat[i];
        norm_y += y[i] * y[i];
        norm_yHat += yHat[i] * yHat[i];
    }
    norm_y = sqrtf(norm_y);
    norm_yHat = sqrtf(norm_yHat);
    if (norm_y == 0 || norm_yHat == 0)
    {
        fprintf(stderr, "[cosine_similarity_loss_derivative] Error: Zero vector norm.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float derivative = -y[0] / (norm_y * norm_yHat);
    for (int i = 1; i < n; i++)
    {
        derivative += (dot_product / (norm_yHat * norm_yHat * norm_yHat)) * yHat[i];
    }
    return derivative;
}
