#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/mean_squared_error.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

/**
 * @brief Computes the Mean Squared Error (MSE).
 *
 * The MSE is defined as:
 * - error = 1/n * Σ (y - yHat)^2
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed error, or an error code if inputs are invalid.
 */
float mean_squared_error(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[mean_squared_error] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }

    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += powf(y[i] - yHat[i], 2);
    }
#if DEBUG_LOGGING
    printf("[mean_squared_error] Computed error: %f\n", sum / n);
#endif
    return sum / n;
}
