/**
 * @file checkpointing.c
 * @brief Gradient checkpointing implementation
 */

#include "autograd/checkpointing.h"
#include "autograd/autograd.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

static bool checkpointing_enabled = false;

typedef struct CheckpointedTensor {
    Tensor* tensor;
    struct IRNode* saved_ir_node; // Save IR node to recompute
    CMLGraph_t saved_ir_context;  // Save IR context
    Tensor** saved_inputs;        // Save inputs for recomputation
    int num_inputs;
} CheckpointedTensor;

// Simple registry for checkpointed tensors (in a full implementation, use a hash table)
static CheckpointedTensor** checkpointed_tensors = NULL;
static int num_checkpointed                      = 0;
static int checkpointed_capacity                 = 0;

void autograd_set_checkpointing(bool enabled) {
    checkpointing_enabled = enabled;
    LOG_DEBUG("Gradient checkpointing %s", enabled ? "enabled" : "disabled");
}

bool autograd_is_checkpointing_enabled(void) { return checkpointing_enabled; }

int autograd_checkpoint(Tensor* tensor) {
    if (!tensor || !checkpointing_enabled)
        return -1;

    // Resize array if needed
    if (num_checkpointed >= checkpointed_capacity) {
        int new_capacity = checkpointed_capacity == 0 ? 16 : checkpointed_capacity * 2;
        CheckpointedTensor** new_array =
            realloc(checkpointed_tensors, (size_t)new_capacity * sizeof(CheckpointedTensor*));
        if (!new_array)
            return -1;

        checkpointed_tensors  = new_array;
        checkpointed_capacity = new_capacity;
    }

    // Create checkpoint entry
    CheckpointedTensor* checkpoint = malloc(sizeof(CheckpointedTensor));
    if (!checkpoint)
        return -1;

    checkpoint->tensor           = tensor;
    checkpoint->saved_ir_node    = tensor->ir_node;
    checkpoint->saved_ir_context = tensor->ir_context;

    // Save inputs if IR node exists
    if (tensor->ir_node && tensor->ir_node->inputs) {
        checkpoint->num_inputs   = tensor->ir_node->num_inputs;
        checkpoint->saved_inputs = malloc((size_t)checkpoint->num_inputs * sizeof(Tensor*));
        if (!checkpoint->saved_inputs) {
            free(checkpoint);
            return -1;
        }
        for (int i = 0; i < checkpoint->num_inputs; i++) {
            checkpoint->saved_inputs[i] = tensor->ir_node->inputs[i];
        }
    } else {
        checkpoint->num_inputs   = 0;
        checkpoint->saved_inputs = NULL;
    }

    // Clear IR node to force recomputation
    tensor->ir_node    = NULL;
    tensor->ir_context = NULL;

    checkpointed_tensors[num_checkpointed] = checkpoint;
    num_checkpointed++;

    LOG_DEBUG("Checkpointed tensor %p", (void*)tensor);
    return 0;
}

Tensor* autograd_recompute(Tensor* tensor) {
    if (!tensor || !checkpointing_enabled)
        return NULL;

    // Find checkpoint entry
    CheckpointedTensor* checkpoint = NULL;
    for (int i = 0; i < num_checkpointed; i++) {
        if (checkpointed_tensors[i] && checkpointed_tensors[i]->tensor == tensor) {
            checkpoint = checkpointed_tensors[i];
            break;
        }
    }

    if (!checkpoint || !checkpoint->saved_ir_node) {
        // Not checkpointed or no saved IR node
        LOG_DEBUG("Tensor %p not checkpointed or no saved IR node", (void*)tensor);
        return tensor;
    }

    LOG_DEBUG("Recomputing tensor %p using IR", (void*)tensor);

    // Attempt to recompute forward pass using saved inputs and IR node
    if (checkpoint->saved_inputs && checkpoint->num_inputs > 0 && checkpoint->saved_ir_node) {
        struct IRNode* node = checkpoint->saved_ir_node;
        UOpType uop_type    = node->type;

        // Recursively recompute inputs if they are also checkpointed
        Tensor** inputs = checkpoint->saved_inputs;
        for (int i = 0; i < checkpoint->num_inputs; i++) {
            if (inputs[i] && !inputs[i]->ir_node) {
                // Input might be checkpointed, try to recompute it
                Tensor* recomputed_input = autograd_recompute(inputs[i]);
                if (recomputed_input && recomputed_input != inputs[i]) {
                    // Input was recomputed, update our reference
                    inputs[i] = recomputed_input;
                }
            }
        }

        // Recompute based on UOpType using uop_* functions
        Tensor* recomputed = NULL;

        switch (uop_type) {
        case UOP_ADD:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_add(inputs[0], inputs[1]);
            }
            break;
        case UOP_SUB:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_sub(inputs[0], inputs[1]);
            }
            break;
        case UOP_MUL:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_mul(inputs[0], inputs[1]);
            }
            break;
        case UOP_DIV:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_div(inputs[0], inputs[1]);
            }
            break;
        case UOP_NEG:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_neg(inputs[0]);
            }
            break;
        case UOP_EXP:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_exp(inputs[0]);
            }
            break;
        case UOP_LOG:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_log(inputs[0]);
            }
            break;
        case UOP_SQRT:
            if (checkpoint->num_inputs == 1) {
                recomputed = uop_sqrt(inputs[0]);
            }
            break;
        case UOP_MATMUL:
            if (checkpoint->num_inputs == 2) {
                recomputed = uop_matmul(inputs[0], inputs[1]);
            }
            break;
        case UOP_SUM:
            if (checkpoint->num_inputs == 1) {
                ReduceParams params = {0};
                params.dims         = NULL;
                params.num_dims     = 0;
                params.keepdim      = false;
                recomputed          = uop_sum(inputs[0], &params);
            }
            break;
        case UOP_MEAN:
            if (checkpoint->num_inputs == 1) {
                ReduceParams params = {0};
                params.dims         = NULL;
                params.num_dims     = 0;
                params.keepdim      = false;
                recomputed          = uop_mean(inputs[0], &params);
            }
            break;
        case UOP_MAX:
        case UOP_CMPLT:
        case UOP_RECIP:
        case UOP_ABS:
        case UOP_SIN:
        case UOP_COS:
        case UOP_TAN:
        case UOP_POW:
        case UOP_MAX_REDUCE:
        case UOP_RESHAPE:
        case UOP_PERMUTE:
        case UOP_EXPAND:
        case UOP_STRIDE:
        case UOP_SLICE:
        case UOP_CONV2D:
        case UOP_WHERE:
        case UOP_COUNT:
            // Not implemented for checkpointing yet
            LOG_DEBUG("UOpType %d not yet supported for recomputation", uop_type);
            break;
        default:
            // For unsupported operations, restore IR node only
            LOG_DEBUG("UOpType %d not supported for recomputation, restoring IR node only",
                      uop_type);
            break;
        }

        if (recomputed) {
            // Update tensor data with recomputed values
            if (tensor->numel == recomputed->numel && tensor->dtype == recomputed->dtype) {
                void* tensor_data     = tensor_data_ptr(tensor);
                void* recomputed_data = tensor_data_ptr(recomputed);
                if (tensor_data && recomputed_data) {
                    size_t data_size = tensor->numel * cml_dtype_size(tensor->dtype);
                    memcpy(tensor_data, recomputed_data, data_size);
                    LOG_DEBUG("Recomputed tensor %p using forward pass", (void*)tensor);
                }
            }

            // Restore IR node for backward pass
            tensor->ir_node    = checkpoint->saved_ir_node;
            tensor->ir_context = checkpoint->saved_ir_context;

            // Free recomputed tensor (we've copied the data)
            tensor_free(recomputed);
        } else {
            // Fallback: restore IR node only
            tensor->ir_node    = checkpoint->saved_ir_node;
            tensor->ir_context = checkpoint->saved_ir_context;
            LOG_DEBUG("Restored IR node for tensor %p (recomputation not available for UOpType %d)",
                      (void*)tensor, uop_type);
        }
    } else {
        // No saved inputs, just restore IR node
        if (checkpoint->saved_ir_node) {
            tensor->ir_node    = checkpoint->saved_ir_node;
            tensor->ir_context = checkpoint->saved_ir_context;
            LOG_DEBUG("Restored IR node for tensor %p (no saved inputs)", (void*)tensor);
        }
    }

    return tensor;
}

// Cleanup function (call during autograd cleanup)
void autograd_checkpointing_cleanup(void) {
    if (checkpointed_tensors) {
        for (int i = 0; i < num_checkpointed; i++) {
            if (checkpointed_tensors[i]) {
                if (checkpointed_tensors[i]->saved_inputs) {
                    free(checkpointed_tensors[i]->saved_inputs);
                }
                free(checkpointed_tensors[i]);
            }
        }
        free(checkpointed_tensors);
        checkpointed_tensors  = NULL;
        num_checkpointed      = 0;
        checkpointed_capacity = 0;
    }
}
