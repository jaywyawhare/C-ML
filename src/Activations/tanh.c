#include "../../include/Activations/tanh.h"
#include "../../include/Core/autograd.h"
#include <math.h>

#define TANH_THRESHOLD 20.0f

float tanH(float x)
{
    if (validate_activation_input(x))
        return 0.0f;
    if (x > TANH_THRESHOLD)
        return 1.0f;
    if (x < -TANH_THRESHOLD)
        return -1.0f;
    float e_pos = expf(x);
    float e_neg = expf(-x);
    return (e_pos - e_neg) / (e_pos + e_neg);
}

Node *tanh_node(Node *x)
{
    if (!x)
        return NULL;
    float result = tanH(x->tensor->storage->data[0]);
    Node *output = tensor(result, x->requires_grad);
    create_activation_node(output, x, OP_TANH, tensor(result, 0));
    return output;
}

void tanh_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0]->requires_grad)
        return;

    SavedVariable *saved = get_saved_variable(inputs[0], 0);
    if (!saved)
        return;

    float tanh_val = saved->tensor->value;
    float grad = grad_output * (1.0f - tanh_val * tanh_val);
    accumulate_grad(inputs[0], grad);
}
