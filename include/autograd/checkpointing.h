#ifndef CML_AUTOGRAD_CHECKPOINTING_H
#define CML_AUTOGRAD_CHECKPOINTING_H

#include "tensor/tensor.h"
#include <stdbool.h>

struct Module;
typedef struct Module Module;
struct Sequential;
typedef struct Sequential Sequential;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * When enabled, activations are recomputed during backward pass
 * instead of being stored, trading computation for memory.
 */
void autograd_set_checkpointing(bool enabled);
bool autograd_is_checkpointing_enabled(void);
int autograd_checkpoint(Tensor* tensor);
Tensor* autograd_recompute(Tensor* tensor);
void autograd_checkpointing_cleanup(void);

typedef struct CheckpointConfig {
    bool enabled;             // Whether checkpointing is active
    int checkpoint_every_n;   // Checkpoint every N layers in sequential models
} CheckpointConfig;

Tensor* checkpoint_forward(Module* module, Tensor* input);
void sequential_apply_checkpointing(Sequential* seq, int every_n);

#ifdef __cplusplus
}
#endif

#endif // CML_AUTOGRAD_CHECKPOINTING_H
