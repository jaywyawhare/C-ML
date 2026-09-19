#ifndef CML_DATA_PARALLEL_H
#define CML_DATA_PARALLEL_H

#include "distributed/distributed.h"
#include "nn.h"
#include "optim.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t bucket_size_bytes;    /* Gradient bucket size (default: 25MB) */
    bool broadcast_buffers;      /* Honored: buffers broadcast from rank 0 at forward */
    bool find_unused_parameters; /* Honored: unused params reserve a zero-filled
                                  * bucket slot so every rank's layout matches */
    /* Accepted for API familiarity but NOT yet honored: gradients are always
     * copied into buckets rather than aliased into them. Inert (not silently
     * wrong) until implemented. */
    int gradient_as_bucket_view; /* NOT YET HONORED */
} DDPConfig;

typedef struct CMLDataParallel {
    Module* module;          /* Wrapped module (not owned) */
    DistProcessGroup* group; /* Process group */
    DDPConfig config;        /* Configuration */

    /* Gradient buckets */
    float** buckets;      /* Bucket buffers */
    int num_buckets;      /* Number of buckets */
    size_t* bucket_sizes; /* Size of each bucket in floats */

    /* Parameter tracking */
    Parameter** all_params; /* All parameters */
    int num_params;         /* Number of parameters */
    int* param_to_bucket;   /* Map param index -> bucket index */

    bool initialized;
} CMLDataParallel;

DDPConfig cml_ddp_default_config(void);

/* Broadcasts parameters from rank 0, sets up gradient buckets. */
CMLDataParallel* cml_ddp_create(Module* module, const DDPConfig* config);

Tensor* cml_ddp_forward(CMLDataParallel* ddp, Tensor* input);

/* Slice a global batch along dim 0 into this rank's shard (extra rows go to the
 * lowest ranks when world_size doesn't divide the batch). Returns a fresh tensor
 * the caller owns; returns `full_batch` unchanged at world_size==1. Feed each
 * rank its shard so data parallelism trains on distinct data rather than N
 * identical replicas. */
Tensor* cml_ddp_shard_input(CMLDataParallel* ddp, Tensor* full_batch);

/* Bucketed all-reduce of gradients, averaged by world_size.
 * Call after tensor_backward() and before optimizer_step(). */
int cml_ddp_sync_gradients(CMLDataParallel* ddp);

/* Does NOT free the underlying module. */
void cml_ddp_free(CMLDataParallel* ddp);

#ifdef __cplusplus
}
#endif

#endif /* CML_DATA_PARALLEL_H */
