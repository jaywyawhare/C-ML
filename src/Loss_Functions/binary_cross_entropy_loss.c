#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/binary_cross_entropy_loss.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

/**
 * @brief Computes the Binary Cross-Entropy Loss.
 *
 * The Binary Cross-Entropy Loss is defined as:
 * - loss = -1/n * Σ [y * log(yHat) + (1 - y) * log(1 - yHat)]
 *
 * @param yHat Pointer to the predicted probabilities.
 * @param y Pointer to the ground truth labels.
 * @param size The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float binary_cross_entropy_loss(float *yHat, float *y, int size)
{
    if (!yHat || !y || size <= 0)
    {
        fprintf(stderr, "[binary_cross_entropy_loss] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }

    float loss = 0.0;

    for (int i = 0; i < size; ++i)
    {

        float epsilon = 1e-15;
        float predicted = fmax(fmin(yHat[i], 1 - epsilon), epsilon);
        loss += -(y[i] * log(predicted) + (1 - y[i]) * log(1 - predicted));
    }

    loss /= size;
#if DEBUG_LOGGING
    printf("[binary_cross_entropy_loss] Computed loss: %f\n", loss);
#endif
    return loss;
}
