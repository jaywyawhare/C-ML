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
    bool broadcast_buffers;      /* Broadcast non-parameter buffers */
    bool find_unused_parameters; /* Find and skip unused params */
    int gradient_as_bucket_view; /* Use gradient views (memory efficient) */
} DDPConfig;

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

DDPConfig cml_ddp_default_config(void);

/* Broadcasts parameters from rank 0, sets up gradient buckets. */
CMLDataParallel* cml_ddp_create(Module* module, const DDPConfig* config);

Tensor* cml_ddp_forward(CMLDataParallel* ddp, Tensor* input);

/* Bucketed all-reduce of gradients, averaged by world_size.
 * Call after tensor_backward() and before optimizer_step(). */
int cml_ddp_sync_gradients(CMLDataParallel* ddp);

/* Does NOT free the underlying module. */
void cml_ddp_free(CMLDataParallel* ddp);

#ifdef __cplusplus
}
#endif

#endif /* CML_DATA_PARALLEL_H */
