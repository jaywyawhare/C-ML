#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/kld_loss.h"
#include "../../include/Core/error_codes.h"

#define EPSILON 1e-8

/**
 * @brief Computes the Kullback-Leibler Divergence Loss.
 *
 * The KLD loss is defined as:
 * - loss = 1/n * Σ [p * log(p / q)]
 *
 * @param p Pointer to the true distribution.
 * @param q Pointer to the predicted distribution.
 * @param n The number of elements in p and q.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float kld_loss(float *p, float *q, int n)
{
    if (!p || !q || n <= 0)
    {
        fprintf(stderr, "[kld_loss] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float p_val = fmaxf(p[i], EPSILON);
        float q_val = fmaxf(q[i], EPSILON);
        sum += p_val * logf(p_val / q_val);
    }
    return sum / n;
}

/**
 * @brief Computes the derivative of the Kullback-Leibler Divergence Loss.
 *
 * The derivative is defined as:
 * - d(loss)/dq = -p / q
 *
 * @param p True distribution value.
 * @param q Predicted distribution value.
 * @return The derivative value.
 */
float kld_loss_derivative(float p, float q)
{
    return -p / fmaxf(q, EPSILON);
}
