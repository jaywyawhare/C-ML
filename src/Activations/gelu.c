#include "../../include/Activations/gelu.h"
#include "../../include/Core/autograd.h"
#include <math.h>

float gelu(float x)
{
    if (validate_activation_input(x))
        return 0.0f;
    const float sqrt_2_over_pi = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

Node *gelu_node(Node *x)
{
    if (!x)
        return NULL;
    float result = gelu(x->tensor->storage->data[0]);
    Node *output = tensor(result, x->requires_grad);
    create_activation_node(output, x, OP_GELU, x);
    return output;
}

void gelu_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0]->requires_grad)
        return;

    float x = inputs[0]->tensor->storage->data[0];
    const float sqrt_2_over_pi = 0.7978845608f;
    float term = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    float tanh_term = tanhf(term);

    float grad = grad_output * 0.5f * (1.0f + tanh_term + x * (sqrt_2_over_pi * (1.0f - tanh_term * tanh_term) * (1.0f + 0.134145f * x * x)));
    accumulate_grad(inputs[0], grad);
}