#include "../../include/Activations/elu.h"
#include "../../include/Core/memory_management.h"
#include <math.h>
#include <stddef.h>

float elu_scalar(float x, float alpha)
{
    if (validate_activation_input(x))
        return 0.0f;
    return x >= 0 ? x : alpha * (expf(x) - 1);
}

Node *elu_node(Node *x, float alpha)
{
    if (!x)
        return NULL;
    float result = elu_scalar(x->tensor->storage->data[0], alpha);
    Node *output = create_node_with_value(result, x->requires_grad);
    
    if (x->requires_grad) {
        // Store alpha value and setup gradient function
        output->grad_fn = (Function *)cm_safe_malloc(sizeof(Function), __FILE__, __LINE__);
        output->grad_fn->op_type = OP_ELU;
        output->grad_fn->forward = NULL;
        output->grad_fn->backward = NULL;  // Will use different backward mechanism
        output->grad_fn->alpha = alpha;
        set_graph_dependencies(output, &x, 1);
    }
    
    return output;
}

void elu_backward(Node *grad_output, Node *node)
{
    if (!node->input_nodes || node->num_inputs == 0)
        return;

    Node *input = node->input_nodes[0];
    if (!input->requires_grad)
        return;

    float x = input->tensor->storage->data[0];
    float alpha = node->grad_fn->alpha;
    float grad_val = x >= 0 ? grad_output->tensor->storage->data[0] : 
                         grad_output->tensor->storage->data[0] * alpha * expf(x);
    
    // Accumulate gradient using the float field
    input->grad += grad_val;
}