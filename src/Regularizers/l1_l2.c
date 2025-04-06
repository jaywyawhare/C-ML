#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../include/Core/error_codes.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

/**
 * @brief Applies combined L1 and L2 regularization to update gradients.
 *
 * The combined regularization adds both the absolute and squared values of weights
 * to the loss function and updates the gradients accordingly.
 *
 * @param w Pointer to the weights array.
 * @param dw Pointer to the gradients array.
 * @param l1 Regularization strength for L1.
 * @param l2 Regularization strength for L2.
 * @param n Number of weights.
 * @return The computed loss or an error code.
 */
float l1_l2(float *w, float *dw, float l1, float l2, int n)
{
    if (w == NULL || dw == NULL)
    {
        fprintf(stderr, "[l1_l2] Error: Null pointer argument.\n");
        return CM_NULL_POINTER_ERROR;
    }
    if (n <= 0)
    {
        fprintf(stderr, "[l1_l2] Error: length of weights must be positive.\n");
        return CM_INVALID_PARAMETER_ERROR;
    }

    float loss = 0;
    for (int i = 0; i < n; i++)
    {
        if (isnan(w[i]) || isinf(w[i]) || isnan(l1) || isinf(l1) || isnan(l2) || isinf(l2))
        {
            fprintf(stderr, "[l1_l2] Error: Invalid parameter(s) provided.\n");
            return CM_INVALID_INPUT_ERROR;
        }
        loss += l1 * fabs(w[i]) + l2 * pow(w[i], 2);

        float l1_grad = (w[i] > 0) ? 1 : (w[i] < 0) ? -1
                                                    : 0;

        dw[i] += l1 * l1_grad + 2 * l2 * w[i];

#if DEBUG_LOGGING
        printf("[l1_l2] Debug: i: %d, w[i]: %f, dw[i]: %f, loss: %f\n", i, w[i], dw[i], loss);
#endif
    }
    return loss;
}