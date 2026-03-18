#ifndef CML_DISTRIBUTED_H
#define CML_DISTRIBUTED_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DIST_BACKEND_NCCL = 0,  /* NVIDIA NCCL (GPU) */
    DIST_BACKEND_MPI,       /* MPI (CPU/GPU) */
    DIST_BACKEND_GLOO,      /* Gloo (CPU fallback) */
    DIST_BACKEND_COUNT
} DistBackendType;

typedef enum {
    DIST_REDUCE_SUM = 0,
    DIST_REDUCE_PRODUCT,
    DIST_REDUCE_MAX,
    DIST_REDUCE_MIN,
    DIST_REDUCE_AVG
} DistReduceOp;

typedef struct DistWork {
    void* internal;       /* Backend-specific handle */
    bool completed;       /* Whether operation is done */
    int error_code;       /* 0 on success */
} DistWork;

typedef struct DistCommOps {
    int (*allreduce)(Tensor* tensor, DistReduceOp op, void* ctx);
    int (*broadcast)(Tensor* tensor, int src_rank, void* ctx);
    int (*allgather)(Tensor** output, Tensor* input, void* ctx);
    int (*reduce_scatter)(Tensor* output, Tensor* input, DistReduceOp op, void* ctx);
    int (*barrier)(void* ctx);
    int (*send)(Tensor* tensor, int dst_rank, int tag, void* ctx);
    int (*recv)(Tensor* tensor, int src_rank, int tag, void* ctx);

    /* Async variants */
    DistWork* (*allreduce_async)(Tensor* tensor, DistReduceOp op, void* ctx);
    int (*wait)(DistWork* work);

    /* Lifecycle */
    int (*init)(void* ctx, int world_size, int rank);
    void (*destroy)(void* ctx);

    /* Backend-owned context, passed as ctx to all ops above */
    void* backend_ctx;
} DistCommOps;

typedef struct DistProcessGroup {
    int rank;               /* This process's rank */
    int world_size;         /* Total number of processes */
    DistBackendType backend;/* Communication backend */
    DistCommOps* ops;       /* Backend operations */
    void* backend_ctx;      /* Backend-specific context */
    bool initialized;       /* Whether group is initialized */
} DistProcessGroup;

int cml_dist_init(DistBackendType backend, int world_size, int rank);

DistProcessGroup* cml_dist_get_default_group(void);

int cml_dist_get_rank(void);

int cml_dist_get_world_size(void);

bool cml_dist_is_initialized(void);

void cml_dist_destroy(void);

int cml_dist_allreduce(Tensor* tensor, DistReduceOp op);

int cml_dist_broadcast(Tensor* tensor, int src_rank);

int cml_dist_allgather(Tensor** output, Tensor* input);

int cml_dist_barrier(void);

DistWork* cml_dist_allreduce_async(Tensor* tensor, DistReduceOp op);

int cml_dist_wait(DistWork* work);

void cml_dist_work_free(DistWork* work);

#ifdef __cplusplus
}
#endif

#endif /* CML_DISTRIBUTED_H */
