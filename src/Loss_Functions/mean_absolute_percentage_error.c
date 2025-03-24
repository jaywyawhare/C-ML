#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/mean_absolute_percentage_error.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

/**
 * @brief Computes the Mean Absolute Percentage Error (MAPE).
 *
 * The MAPE is defined as:
 * - error = 1/n * Σ |(y - yHat) / y| * 100
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed error as a percentage, or an error code if inputs are invalid.
 */
float mean_absolute_percentage_error(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[mean_absolute_percentage_error] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }

    float sum = 0;
    int valid_count = 0;
    for (int i = 0; i < n; i++)
    {
        if (fabsf(y[i]) < 1e-15)
        {
            continue;
        }
        sum += fabsf((y[i] - yHat[i]) / y[i]);
        valid_count++;
    }
    if (valid_count == 0)
    {
        return 0;
    }
#if DEBUG_LOGGING
    printf("[mean_absolute_percentage_error] Computed error: %f\n", sum / valid_count * 100);
#endif
    return sum / valid_count * 100;
}
