#include "../../include/Activations/sigmoid.h"
#include "../../include/Core/autograd.h"
#include <math.h>

float sigmoid(float x)
{
    if (validate_activation_input(x))
        return 0.0f;
    return x >= 0 ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}

Node *sigmoid_node(Node *x)
{
    if (!x)
        return NULL;
    float result = sigmoid(x->tensor->storage->data[0]);
    Node *output = tensor(result, x->requires_grad);
    create_activation_node(output, x, OP_SIGMOID, tensor(result, 0));
    return output;
}

void sigmoid_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0]->requires_grad)
        return;

    SavedVariable *saved = get_saved_variable(inputs[0], 0);
    if (!saved)
        return;

    float sig = saved->tensor->value;
    float grad = grad_output * sig * (1.0f - sig);
    accumulate_grad(inputs[0], grad);
}