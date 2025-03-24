#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Activations/softmax.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

/**
 * @brief Applies the softmax activation function.
 *
 * The softmax function is defined as:
 * - softmax(z_i) = exp(z_i) / sum(exp(z_j)) for j = 1 to n
 *
 * @param z Pointer to the input array.
 * @param n The number of elements in the input array.
 * @return Pointer to the output array containing the softmax values.
 */
float *softmax(float *z, int n)
{
    if (z == NULL || n <= 0)
    {
        fprintf(stderr, "[softmax] Error: Null input pointer or invalid size (%d).\n", n);
        return (float *)CM_NULL_POINTER_ERROR;
    }

    for (int i = 0; i < n; i++)
    {
        if (isnan(z[i]) || isinf(z[i]))
        {
            fprintf(stderr, "[softmax] Error: Invalid input value (NaN or Inf) at index %d.\n", i);
            return (float *)CM_INVALID_INPUT_ERROR;
        }
    }

    float max_val = z[0];
    for (int i = 1; i < n; i++)
    {
        if (z[i] > max_val)
        {
            max_val = z[i];
        }
    }

    float sum = 0.0f;
    float *output = cm_safe_malloc(n * sizeof(float), __FILE__, __LINE__);
    if (output == NULL)
    {
        fprintf(stderr, "[softmax] Error: Memory allocation failed.\n");
        return (float *)CM_MEMORY_ALLOCATION_ERROR;
    }

    for (int i = 0; i < n; i++)
    {
        output[i] = expf(z[i] - max_val);
        sum += output[i];
    }

    if (sum == 0.0f)
    {
        fprintf(stderr, "[softmax] Error: Division by zero (sum of exponentials is zero).\n");
        cm_safe_free((void **)&output);
        return (float *)CM_DIVISION_BY_ZERO_ERROR;
    }

    for (int i = 0; i < n; i++)
    {
        output[i] /= sum;
#if DEBUG_LOGGING
        printf("[softmax] Output[%d]: %f\n", i, output[i]);
#endif
    }

    return output;
}

/**
 * @brief Frees the memory allocated for the softmax output.
 *
 * @param output Pointer to the output array to be freed.
 */
void freeSoftmax(float **output)
{
    if (output != NULL && *output != NULL)
    {
        cm_safe_free((void **)output);
#if DEBUG_LOGGING
        printf("[freeSoftmax] Memory freed for softmax output.\n");
#endif
    }
}

/**
 * @brief Computes the derivative of the softmax activation function.
 *
 * The derivative of softmax is:
 * - f'(z_i) = f(z_i) * (1 - f(z_i)) for diagonal elements
 * - f'(z_i, z_j) = -f(z_i) * f(z_j) for off-diagonal elements
 *
 * @param softmax_output Pointer to the softmax output array.
 * @param n The number of elements in the output array.
 * @param jacobian Pointer to the output Jacobian matrix (n x n).
 * @return Error code indicating success or failure.
 */
int softmax_derivative(float *softmax_output, int n, float *jacobian)
{
    if (softmax_output == NULL || jacobian == NULL || n <= 0)
    {
        fprintf(stderr, "[softmax_derivative] Error: Null pointer or invalid size.\n");
        return CM_NULL_POINTER_ERROR;
    }

    for (int i = 0; i < n; i++)
    {
        if (isnan(softmax_output[i]) || isinf(softmax_output[i]) || softmax_output[i] < 0.0f || softmax_output[i] > 1.0f)
        {
            fprintf(stderr, "[softmax_derivative] Error: Invalid softmax output at index %d.\n", i);
            return CM_INVALID_INPUT_ERROR;
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                jacobian[i * n + j] = softmax_output[i] * (1.0f - softmax_output[i]);
            }
            else
            {
                jacobian[i * n + j] = -softmax_output[i] * softmax_output[j];
            }
        }
    }

    return CM_SUCCESS;
}
