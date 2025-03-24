#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/mean_absolute_error.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

/**
 * @brief Computes the Mean Absolute Error (MAE).
 *
 * The MAE is defined as:
 * - error = 1/n * Σ |y - yHat|
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed error, or an error code if inputs are invalid.
 */
float mean_absolute_error(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[mean_absolute_error] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }

    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += fabsf(y[i] - yHat[i]);
    }
#if DEBUG_LOGGING
    printf("[mean_absolute_error] Computed error: %f\n", sum / n);
#endif
    return sum / n;
}
