#ifndef CML_AUTOGRAD_CHECKPOINTING_H
#define CML_AUTOGRAD_CHECKPOINTING_H

#include "tensor/tensor.h"
#include <stdbool.h>

// Forward declarations to avoid circular dependency
struct Module;
typedef struct Module Module;
struct Sequential;
typedef struct Sequential Sequential;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enable gradient checkpointing
 *
 * When enabled, activations are recomputed during backward pass
 * instead of being stored, trading computation for memory.
 *
 * @param enabled Enable flag
 */
void autograd_set_checkpointing(bool enabled);

/**
 * @brief Check if gradient checkpointing is enabled
 *
 * @return true if enabled, false otherwise
 */
bool autograd_is_checkpointing_enabled(void);

/**
 * @brief Checkpoint a tensor for gradient checkpointing
 *
 * Marks a tensor to be recomputed during backward pass
 * instead of storing its activation.
 *
 * @param tensor Tensor to checkpoint
 * @return 0 on success, negative on failure
 */
int autograd_checkpoint(Tensor* tensor);

/**
 * @brief Recompute tensor during backward pass (if checkpointed)
 *
 * @param tensor Tensor to recompute
 * @return Recomputed tensor, or NULL on failure
 */
Tensor* autograd_recompute(Tensor* tensor);

/**
 * @brief Cleanup checkpointing state
 *
 * Frees all checkpointed tensors and resets state.
 * Should be called during autograd shutdown.
 */
void autograd_checkpointing_cleanup(void);

/**
 * @brief Checkpoint configuration for training integration
 */
typedef struct CheckpointConfig {
    bool enabled;             // Whether checkpointing is active
    int checkpoint_every_n;   // Checkpoint every N layers in sequential models
} CheckpointConfig;

/**
 * @brief Run forward pass through a module with checkpointing
 *
 * Runs the forward pass normally but marks the output for recomputation
 * during the backward pass, saving memory at the cost of extra computation.
 *
 * @param module Module to run
 * @param input Input tensor
 * @return Output tensor (marked for recomputation during backward)
 */
Tensor* checkpoint_forward(Module* module, Tensor* input);

/**
 * @brief Apply checkpointing to a Sequential model
 *
 * Configures the sequential model to checkpoint every N layers,
 * reducing peak memory usage during training.
 *
 * @param seq Sequential model
 * @param every_n Checkpoint every N layers (0 to disable)
 */
void sequential_apply_checkpointing(Sequential* seq, int every_n);

#ifdef __cplusplus
}
#endif

#endif // CML_AUTOGRAD_CHECKPOINTING_H
