#include "../../include/Activations/leaky_relu.h"
#include "../../include/Core/autograd.h"
#include <math.h>

#define LEAKY_RELU_ALPHA 0.01f

float leaky_relu(float x)
{
   if (validate_activation_input(x))
      return 0.0f;
   return x > 0 ? x : LEAKY_RELU_ALPHA * x;
}

Node *leaky_relu_node(Node *x)
{
   if (!x)
      return NULL;
   float result = leaky_relu(x->tensor->storage->data[0]);
   Node *output = tensor(result, x->requires_grad);
   create_activation_node(output, x, OP_LEAKY_RELU, NULL);
   return output;
}

void leaky_relu_backward(float grad_output, Node **inputs, int ninputs)
{
   if (ninputs != 1 || !inputs[0]->requires_grad)
      return;

   float x = inputs[0]->tensor->storage->data[0];
   float grad = x > 0 ? grad_output : LEAKY_RELU_ALPHA * grad_output;
   accumulate_grad(inputs[0], grad);
}