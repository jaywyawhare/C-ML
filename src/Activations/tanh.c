#include "../../include/Activations/tanh.h"
#include <math.h>

#define TANH_THRESHOLD 20.0f

float tanH_scalar(float x)
{
    if (x > TANH_THRESHOLD)
        return 1.0f;
    if (x < -TANH_THRESHOLD)
        return -1.0f;
    float e_pos = expf(x);
    float e_neg = expf(-x);
    return (e_pos - e_neg) / (e_pos + e_neg);
}

void tanh_backward(float grad_output, Node **inputs, int ninputs)
{
    // Placeholder implementation for backward pass
    // The actual backward computation is handled by the autograd system
}
