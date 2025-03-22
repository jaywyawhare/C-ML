#include <math.h>
#include "../../include/Activations/sigmoid.h"

/**
 * @brief Applies the sigmoid activation function.
 *
 * The sigmoid activation function is defined as:
 * - f(x) = 1 / (1 + exp(-x))
 *
 * @param x The input value.
 * @return The result of the sigmoid activation function.
 */

float sigmoid(float x)
{
    if (x >= 0)
    {
        float exp_neg_x = expf(-x);
        return 1 / (1 + exp_neg_x);
    }
    else
    {
        float exp_pos_x = expf(x);
        return exp_pos_x / (1 + exp_pos_x);
    }
}