#include "../../include/Activations/linear.h"
#include "../../include/Core/autograd.h"

float linear(float x)
{
    if (validate_activation_input(x))
        return 0.0f;
    return x;
}

Node *linear_node(Node *x)
{
    if (!x)
        return NULL;
    float result = linear(x->tensor->storage->data[0]);
    Node *output = tensor(result, x->requires_grad);
    create_activation_node(output, x, OP_LINEAR, NULL);
    return output;
}

void linear_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0]->requires_grad)
        return;

    // Linear function derivative is 1
    accumulate_grad(inputs[0], grad_output);
}