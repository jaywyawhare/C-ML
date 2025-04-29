#include "../../include/Activations/relu.h"
#include "../../include/Core/autograd.h"

float relu(float x)
{
    if (validate_activation_input(x))
        return 0.0f;
    return x > 0 ? x : 0;
}

Node *relu_node(Node *x)
{
    if (!x)
        return NULL;
    float result = relu(x->tensor->storage->data[0]);
    Node *output = tensor(result, x->requires_grad);
    create_activation_node(output, x, OP_RELU, x);
    return output;
}

void relu_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0]->requires_grad)
        return;

    Node *input = inputs[0];
    float x = input->tensor->storage->data[0];
    float grad = x > 0 ? grad_output : 0.0f;
    accumulate_grad(input, grad);
}
