/**
 * @file checkpointing.c
 * @brief Gradient checkpointing implementation
 */

#include "autograd/checkpointing.h"
#include "autograd/autograd.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <stdlib.h>

// Global checkpointing state
static bool checkpointing_enabled = false;

// Checkpointed tensor structure
typedef struct CheckpointedTensor {
    Tensor* tensor;
    Function* saved_grad_fn; // Save grad_fn to recompute
    Tensor** saved_inputs;   // Save inputs for recomputation
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
            CM_REALLOC(checkpointed_tensors, new_capacity * sizeof(CheckpointedTensor*));
        if (!new_array)
            return -1;

        checkpointed_tensors  = new_array;
        checkpointed_capacity = new_capacity;
    }

    // Create checkpoint entry
    CheckpointedTensor* checkpoint = CM_MALLOC(sizeof(CheckpointedTensor));
    if (!checkpoint)
        return -1;

    checkpoint->tensor        = tensor;
    checkpoint->saved_grad_fn = tensor->grad_fn;

    // Save inputs if grad_fn exists
    if (tensor->grad_fn && tensor->grad_fn->inputs) {
        checkpoint->num_inputs   = tensor->grad_fn->num_inputs;
        checkpoint->saved_inputs = CM_MALLOC(checkpoint->num_inputs * sizeof(Tensor*));
        if (checkpoint->saved_inputs) {
            for (int i = 0; i < checkpoint->num_inputs; i++) {
                checkpoint->saved_inputs[i] = tensor->grad_fn->inputs[i];
            }
        }
    } else {
        checkpoint->num_inputs   = 0;
        checkpoint->saved_inputs = NULL;
    }

    // Clear grad_fn to force recomputation
    tensor->grad_fn = NULL;

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

    if (!checkpoint || !checkpoint->saved_grad_fn) {
        // TODO
        // Not checkpointed or no saved function
        return tensor;
    }

    LOG_DEBUG("Recomputing tensor %p", (void*)tensor);
    // TODO
    // Recompute forward pass
    // This is a simplified implementation - in a full implementation,
    // we would re-execute the forward function with saved inputs
    if (checkpoint->saved_inputs && checkpoint->num_inputs > 0) {
        // For now, just restore the grad_fn
        // A full implementation would re-execute the forward function
        tensor->grad_fn = checkpoint->saved_grad_fn;
    }

    return tensor;
}

// Cleanup function (call during autograd cleanup)
void autograd_checkpointing_cleanup(void) {
    if (checkpointed_tensors) {
        for (int i = 0; i < num_checkpointed; i++) {
            if (checkpointed_tensors[i]) {
                if (checkpointed_tensors[i]->saved_inputs) {
                    CM_FREE(checkpointed_tensors[i]->saved_inputs);
                }
                CM_FREE(checkpointed_tensors[i]);
            }
        }
        CM_FREE(checkpointed_tensors);
        checkpointed_tensors  = NULL;
        num_checkpointed      = 0;
        checkpointed_capacity = 0;
    }
}
