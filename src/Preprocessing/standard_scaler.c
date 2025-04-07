#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../include/Preprocessing/standard_scaler.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

/**
 * @brief Scales an array of floats to have a mean of 0 and a standard deviation of 1.
 *
 * The function calculates the mean and standard deviation of the input array and
 * scales the values using the formula:
 * scaled_value = (value - mean) / std
 *
 * @param x The input array of floats.
 * @param size The size of the input array.
 * @return A pointer to the scaled array, or NULL if an error occurs.
 */
float *standard_scaler(float *x, int size)
{
    if (x == NULL)
    {
        LOG_ERROR("Null pointer argument");
        return NULL;
    }

    if (size <= 0)
    {
        LOG_ERROR("Invalid size argument");
        return NULL;
    }

    float *scaled = (float *)cm_safe_malloc(sizeof(float) * size, __FILE__, __LINE__);
    if (scaled == NULL)
    {
        LOG_ERROR("Memory allocation failed\n");
        return NULL;
    }

    float mean = 0;
    for (int i = 0; i < size; i++)
    {
        mean += x[i];
    }
    mean /= size;

    float std = 0;
    for (int i = 0; i < size; i++)
    {
        std += pow(x[i] - mean, 2);
    }
    std /= size;
    std = sqrt(std);

    if (std == 0)
    {
        LOG_ERROR("Standard deviation is zero\n");
        free(scaled);
        return NULL;
    }

    for (int i = 0; i < size; i++)
    {
        scaled[i] = (x[i] - mean) / std;
#if DEBUG_LOGGING
        LOG_DEBUG("Scaled[%d]: %f", i, scaled[i]);
#endif
    }
#if DEBUG_LOGGING
    LOG_DEBUG("Scaling complete.");
#endif
    return scaled;
}