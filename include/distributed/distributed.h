/**
 * @file distributed.h
 * @brief Distributed training API - process group, collectives
 *
 * Provides distributed communication primitives for multi-GPU and
 * multi-node training. Supports NCCL, MPI, and Gloo backends.
 */

#ifndef CML_DISTRIBUTED_H
#define CML_DISTRIBUTED_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Communication backend type
 */
typedef enum {
    DIST_BACKEND_NCCL = 0,  /* NVIDIA NCCL (GPU) */
    DIST_BACKEND_MPI,       /* MPI (CPU/GPU) */
    DIST_BACKEND_GLOO,      /* Gloo (CPU fallback) */
    DIST_BACKEND_COUNT
} DistBackendType;

/**
 * @brief Reduction operation type
 */
typedef enum {
    DIST_REDUCE_SUM = 0,
    DIST_REDUCE_PRODUCT,
    DIST_REDUCE_MAX,
    DIST_REDUCE_MIN,
    DIST_REDUCE_AVG
} DistReduceOp;

/**
 * @brief Communication operation handle (for async ops)
 */
typedef struct DistWork {
    void* internal;       /* Backend-specific handle */
    bool completed;       /* Whether operation is done */
    int error_code;       /* 0 on success */
} DistWork;

/**
 * @brief Communication operations vtable
 */
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
} DistCommOps;

/**
 * @brief Process group
 */
typedef struct DistProcessGroup {
    int rank;               /* This process's rank */
    int world_size;         /* Total number of processes */
    DistBackendType backend;/* Communication backend */
    DistCommOps* ops;       /* Backend operations */
    void* backend_ctx;      /* Backend-specific context */
    bool initialized;       /* Whether group is initialized */
} DistProcessGroup;

/**
 * @brief Initialize distributed training
 *
 * @param backend Backend type
 * @param world_size Total number of processes (-1 = auto from env)
 * @param rank This process's rank (-1 = auto from env)
 * @return 0 on success, -1 on failure
 */
int cml_dist_init(DistBackendType backend, int world_size, int rank);

/**
 * @brief Get the default process group
 */
DistProcessGroup* cml_dist_get_default_group(void);

/**
 * @brief Get this process's rank
 */
int cml_dist_get_rank(void);

/**
 * @brief Get world size
 */
int cml_dist_get_world_size(void);

/**
 * @brief Check if distributed is initialized
 */
bool cml_dist_is_initialized(void);

/**
 * @brief Shutdown distributed training
 */
void cml_dist_destroy(void);

/**
 * @brief All-reduce a tensor across all processes
 *
 * @param tensor Tensor to reduce (modified in-place)
 * @param op Reduction operation
 * @return 0 on success
 */
int cml_dist_allreduce(Tensor* tensor, DistReduceOp op);

/**
 * @brief Broadcast a tensor from src_rank to all processes
 *
 * @param tensor Tensor to broadcast (modified in-place on non-src ranks)
 * @param src_rank Source rank
 * @return 0 on success
 */
int cml_dist_broadcast(Tensor* tensor, int src_rank);

/**
 * @brief All-gather tensors from all processes
 *
 * @param output Array of tensors (one per rank, pre-allocated)
 * @param input Local input tensor
 * @return 0 on success
 */
int cml_dist_allgather(Tensor** output, Tensor* input);

/**
 * @brief Barrier synchronization
 */
int cml_dist_barrier(void);

/**
 * @brief Async all-reduce
 *
 * @param tensor Tensor to reduce
 * @param op Reduction operation
 * @return Work handle (call cml_dist_wait to complete)
 */
DistWork* cml_dist_allreduce_async(Tensor* tensor, DistReduceOp op);

/**
 * @brief Wait for async operation to complete
 */
int cml_dist_wait(DistWork* work);

/**
 * @brief Free a work handle
 */
void cml_dist_work_free(DistWork* work);

#ifdef __cplusplus
}
#endif

#endif /* CML_DISTRIBUTED_H */
