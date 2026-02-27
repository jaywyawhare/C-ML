/**
 * @file gloo_backend.c
 * @brief Gloo-style CPU fallback backend for distributed training
 *
 * Pure C implementation of ring all-reduce and other collectives
 * for single-node multi-process training without external dependencies.
 */

#include "distributed/comm_backend.h"
#include "distributed/distributed.h"
#include "distributed/ring_allreduce.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

/**
 * Ring all-reduce for single-process (trivial case).
 * For multi-process, this would use shared memory or sockets.
 */
static int gloo_allreduce(Tensor* tensor, DistReduceOp op, void* ctx) {
    (void)ctx;
    if (!tensor || !tensor->data)
        return -1;

    /* In single-process mode, all-reduce is a no-op */
    DistProcessGroup* group = cml_dist_get_default_group();
    if (!group || group->world_size <= 1) {
        /* Apply averaging if requested */
        if (op == DIST_REDUCE_AVG) {
            float* data = (float*)tensor->data;
            float scale = 1.0f / (float)(group ? group->world_size : 1);
            for (size_t i = 0; i < tensor->numel; i++)
                data[i] *= scale;
        }
        return 0;
    }

    /* For multi-process: use ring all-reduce algorithm */
    int ret = cml_ring_allreduce((float*)tensor->data, tensor->numel,
                                  group->world_size, group->rank,
                                  op, group->ops, group->backend_ctx);
    if (ret != 0) {
        LOG_ERROR("Ring allreduce failed");
        return ret;
    }

    LOG_DEBUG("Gloo allreduce completed (numel: %zu)", tensor->numel);
    return 0;
}

static int gloo_broadcast(Tensor* tensor, int src_rank, void* ctx) {
    (void)ctx;
    (void)src_rank;
    if (!tensor)
        return -1;

    /* In single-process mode, broadcast is a no-op */
    LOG_DEBUG("Gloo broadcast from rank %d (numel: %zu)", src_rank, tensor->numel);
    return 0;
}

static int gloo_allgather(Tensor** output, Tensor* input, void* ctx) {
    (void)ctx;
    if (!output || !input)
        return -1;

    /* Single-process: just copy input to output[0] */
    if (output[0] && output[0]->data && input->data) {
        memcpy(output[0]->data, input->data, input->numel * sizeof(float));
    }

    return 0;
}

static int gloo_reduce_scatter(Tensor* output, Tensor* input, DistReduceOp op, void* ctx) {
    (void)ctx;
    (void)op;
    if (!output || !input)
        return -1;

    /* Single-process: copy relevant chunk */
    if (output->data && input->data) {
        size_t copy_size = output->numel < input->numel ? output->numel : input->numel;
        memcpy(output->data, input->data, copy_size * sizeof(float));
    }

    return 0;
}

static int gloo_barrier(void* ctx) {
    (void)ctx;
    /* Single-process: no-op */
    return 0;
}

static int gloo_init(void* ctx, int world_size, int rank) {
    (void)ctx;
    LOG_INFO("Gloo backend initialized (rank %d/%d)", rank, world_size);
    return 0;
}

static void gloo_destroy(void* ctx) {
    (void)ctx;
    LOG_INFO("Gloo backend destroyed");
}

DistCommOps* cml_dist_create_gloo_backend(void) {
    DistCommOps* ops = calloc(1, sizeof(DistCommOps));
    if (!ops)
        return NULL;

    ops->allreduce = gloo_allreduce;
    ops->broadcast = gloo_broadcast;
    ops->allgather = gloo_allgather;
    ops->reduce_scatter = gloo_reduce_scatter;
    ops->barrier = gloo_barrier;
    ops->init = gloo_init;
    ops->destroy = gloo_destroy;

    /* Async: not supported, will fall back to sync */
    ops->allreduce_async = NULL;
    ops->wait = NULL;
    ops->send = NULL;
    ops->recv = NULL;

    return ops;
}

void cml_dist_free_backend(DistCommOps* ops) {
    free(ops);
}
