/**
 * @file checkpointing.h
 * @brief Gradient checkpointing for memory-efficient backward pass
 */

#ifndef CML_AUTOGRAD_CHECKPOINTING_H
#define CML_AUTOGRAD_CHECKPOINTING_H

#include "tensor/tensor.h"
#include <stdbool.h>

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

#ifdef __cplusplus
}
#endif

#endif // CML_AUTOGRAD_CHECKPOINTING_H
