#include <math.h>
#include <string.h>
#include "../../include/Activations/softmax.h"
#include "../../include/Core/autograd.h"
#include "../../include/Core/memory_management.h"

float *softmax(float *z, int n)
{
    if (!z || n <= 0 || validate_activation_input(z[0]))
        return NULL;

    // Allocate output and find max for numerical stability
    float *output = cm_safe_malloc(n * sizeof(float), __FILE__, __LINE__);
    if (!output)
        return NULL;

    float max_val = z[0];
    float sum = 0.0f;

    // Single pass to find max
    for (int i = 1; i < n; i++)
    {
        max_val = z[i] > max_val ? z[i] : max_val;
    }

    // Calculate exp(x - max) and sum
    for (int i = 0; i < n; i++)
    {
        output[i] = expf(z[i] - max_val);
        sum += output[i];
    }

    if (sum == 0.0f)
    {
        cm_safe_free((void **)&output);
        return NULL;
    }

    // Normalize
    for (int i = 0; i < n; i++)
    {
        output[i] /= sum;
    }

    return output;
}

Node *softmax_node(Node *x)
{
    if (!x)
        return NULL;

    float *result = softmax(x->tensor->storage->data, x->tensor->storage->size);
    if (!result)
        return NULL;

    Node *output = empty_like(x);
    if (!output)
    {
        cm_safe_free((void **)&result);
        return NULL;
    }

    memcpy(output->tensor->storage->data, result, x->tensor->storage->size * sizeof(float));
    cm_safe_free((void **)&result);
    create_activation_node(output, x, OP_SOFTMAX, NULL);
    return output;
}

void softmax_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0]->requires_grad)
        return;

    Node *input = inputs[0];
    size_t n = input->tensor->storage->size;

    // Recompute softmax for backward pass
    float *softmax_vals = softmax(input->tensor->storage->data, n);
    if (!softmax_vals)
        return;

    // Compute Jacobian-vector product
    for (size_t i = 0; i < n; i++)
    {
        float grad_i = 0.0f;
        for (size_t j = 0; j < n; j++)
        {
            float kronecker = (i == j) ? 1.0f : 0.0f;
            grad_i += grad_output * softmax_vals[j] * (kronecker - softmax_vals[i]);
        }

        // Check if this is backward and needs gradient
        if (input->requires_grad && grad_output != 0)
            accumulate_grad(input, grad_i);
    }

    cm_safe_free((void **)&softmax_vals);
}
