/**
 * @file data_parallel.h
 * @brief Distributed Data Parallel (DDP) training
 *
 * Implements PyTorch-style DDP: broadcast params from rank 0 at init,
 * bucketed gradient all-reduce (25MB buckets), average by world_size.
 */

#ifndef CML_DATA_PARALLEL_H
#define CML_DATA_PARALLEL_H

#include "distributed/distributed.h"
#include "nn.h"
#include "optim.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief DDP configuration
 */
typedef struct {
    size_t bucket_size_bytes;    /* Gradient bucket size (default: 25MB) */
    bool broadcast_buffers;      /* Broadcast non-parameter buffers */
    bool find_unused_parameters; /* Find and skip unused params */
    int gradient_as_bucket_view; /* Use gradient views (memory efficient) */
} DDPConfig;

/**
 * @brief DDP wrapper around a module
 */
typedef struct CMLDataParallel {
    Module* module;             /* Wrapped module (not owned) */
    DistProcessGroup* group;    /* Process group */
    DDPConfig config;           /* Configuration */

    /* Gradient buckets */
    float** buckets;            /* Bucket buffers */
    int num_buckets;            /* Number of buckets */
    size_t* bucket_sizes;       /* Size of each bucket in floats */
    bool* bucket_ready;         /* Whether bucket is ready for allreduce */

    /* Parameter tracking */
    Parameter** all_params;     /* All parameters */
    int num_params;             /* Number of parameters */
    int* param_to_bucket;       /* Map param index -> bucket index */

    bool initialized;
} CMLDataParallel;

/**
 * @brief Create default DDP configuration
 */
DDPConfig cml_ddp_default_config(void);

/**
 * @brief Wrap a module for distributed data parallel training
 *
 * Broadcasts parameters from rank 0, sets up gradient buckets.
 *
 * @param module Module to wrap
 * @param config DDP configuration (NULL = defaults)
 * @return DDP wrapper, or NULL on failure
 */
CMLDataParallel* cml_ddp_create(Module* module, const DDPConfig* config);

/**
 * @brief Forward pass through DDP module
 *
 * Same as module_forward, but registers backward hooks for gradient sync.
 *
 * @param ddp DDP wrapper
 * @param input Input tensor
 * @return Output tensor
 */
Tensor* cml_ddp_forward(CMLDataParallel* ddp, Tensor* input);

/**
 * @brief Synchronize gradients after backward pass
 *
 * Performs bucketed all-reduce of gradients and averages by world_size.
 * Call this after tensor_backward() and before optimizer_step().
 *
 * @param ddp DDP wrapper
 * @return 0 on success
 */
int cml_ddp_sync_gradients(CMLDataParallel* ddp);

/**
 * @brief Free DDP wrapper (does NOT free the underlying module)
 */
void cml_ddp_free(CMLDataParallel* ddp);

#ifdef __cplusplus
}
#endif

#endif /* CML_DATA_PARALLEL_H */
