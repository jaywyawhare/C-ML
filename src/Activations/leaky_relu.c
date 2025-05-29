#include "../../include/Activations/leaky_relu.h"
#include <math.h>

#define LEAKY_RELU_ALPHA 0.01f

float leaky_relu_scalar(float x)
{
   return x > 0 ? x : LEAKY_RELU_ALPHA * x;
}

void leaky_relu_backward(float grad_output, Node **inputs, int ninputs)
{
   // Backward function implementation would go here
   // For now this is a placeholder
}