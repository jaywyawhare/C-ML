#include "../../include/Activations/elu.h"
#include "../../include/Core/autograd.h"
#include <math.h>

float elu(float x, float alpha)
{
    if (validate_activation_input(x))
        return 0.0f;
    return x >= 0 ? x : alpha * (expf(x) - 1);
}

Node *elu_node(Node *x, float alpha)
{
    if (!x)
        return NULL;
    float result = elu(x->tensor->storage->data[0], alpha);
    Node *output = tensor(result, x->requires_grad);
    create_activation_node(output, x, OP_ELU, tensor(alpha, 0));
    return output;
}

void elu_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0]->requires_grad)
        return;

    SavedVariable *saved = get_saved_variable(inputs[0], 0);
    if (!saved)
        return;

    float x = inputs[0]->tensor->storage->data[0];
    float alpha = saved->tensor->value;
    float grad = x >= 0 ? grad_output : grad_output * alpha * expf(x);
    accumulate_grad(inputs[0], grad);
}