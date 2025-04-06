#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../include/Preprocessing/min_max_scaler.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

/**
 * @brief Scales an array of floats to a range of [0, 1] using min-max scaling.
 *
 * The function finds the minimum and maximum values in the input array and scales 
 * the values to the range [0, 1] using the formula:
 * scaled_value = (value - min) / (max - min)
 * 
 * @param x The input array of floats.
 * @param size The size of the input array.
 * @return A pointer to the scaled array, or NULL if an error occurs.
 */
float *min_max_scaler(float *x, int size)
{
    if (x == NULL)
    {
        fprintf(stderr, "[minMaxScaler] Error: Null pointer argument\n");
        return NULL;
    }

    if (size <= 0)
    {
        fprintf(stderr, "[minMaxScaler] Error: Invalid size argument\n");
        return NULL;
    }
    float *scaled = (float *)cm_safe_malloc(sizeof(float) * size, __FILE__, __LINE__);
    if (scaled == NULL)
    {
        fprintf(stderr, "[minMaxScaler] Memory allocation failed\n");
        return NULL;
    }
    float min = x[0];
    float max = x[0];
    for (int i = 0; i < size; i++)
    {
        if (x[i] < min)
        {
            min = x[i];
        }
        if (x[i] > max)
        {
            max = x[i];
        }
    }
    if (max == min)
    {
        fprintf(stderr, "[minMaxScaler] Max and min are equal\n");
        free(scaled);
        return NULL;
    }
    for (int i = 0; i < size; i++)
    {
        scaled[i] = (x[i] - min) / (max - min);
#if DEBUG_LOGGING
        printf("[minMaxScaler] Scaled[%d]: %f\n", i, scaled[i]);
#endif
    }
#if DEBUG_LOGGING
    printf("[minMaxScaler] Scaling complete.\n");
#endif
    return scaled;
}