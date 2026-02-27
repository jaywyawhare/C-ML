/**
 * @file distributed.c
 * @brief Process group management and collective operations
 */

#include "distributed/distributed.h"
#include "distributed/comm_backend.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static DistProcessGroup* g_default_group = NULL;

int cml_dist_init(DistBackendType backend, int world_size, int rank) {
    if (g_default_group && g_default_group->initialized) {
        LOG_WARNING("Distributed already initialized");
        return 0;
    }

    /* Auto-detect from environment */
    if (world_size < 0) {
        const char* ws_env = getenv("CML_WORLD_SIZE");
        if (!ws_env) ws_env = getenv("WORLD_SIZE");
        world_size = ws_env ? atoi(ws_env) : 1;
    }

    if (rank < 0) {
        const char* rank_env = getenv("CML_RANK");
        if (!rank_env) rank_env = getenv("RANK");
        if (!rank_env) rank_env = getenv("LOCAL_RANK");
        rank = rank_env ? atoi(rank_env) : 0;
    }

    g_default_group = calloc(1, sizeof(DistProcessGroup));
    if (!g_default_group)
        return -1;

    g_default_group->rank = rank;
    g_default_group->world_size = world_size;
    g_default_group->backend = backend;

    /* Create backend ops */
    DistCommOps* ops = NULL;
    switch (backend) {
    case DIST_BACKEND_NCCL:
        ops = cml_dist_create_nccl_backend();
        if (!ops) {
            LOG_WARNING("NCCL unavailable, falling back to Gloo");
            ops = cml_dist_create_gloo_backend();
            g_default_group->backend = DIST_BACKEND_GLOO;
        }
        break;
    case DIST_BACKEND_MPI:
        ops = cml_dist_create_mpi_backend();
        if (!ops) {
            LOG_WARNING("MPI unavailable, falling back to Gloo");
            ops = cml_dist_create_gloo_backend();
            g_default_group->backend = DIST_BACKEND_GLOO;
        }
        break;
    case DIST_BACKEND_GLOO:
    default:
        ops = cml_dist_create_gloo_backend();
        break;
    }

    if (!ops) {
        LOG_ERROR("Failed to create any communication backend");
        free(g_default_group);
        g_default_group = NULL;
        return -1;
    }

    g_default_group->ops = ops;

    /* Initialize backend */
    if (ops->init) {
        void* ctx = NULL;
        int result = ops->init(ctx, world_size, rank);
        if (result != 0) {
            LOG_ERROR("Backend initialization failed");
            cml_dist_free_backend(ops);
            free(g_default_group);
            g_default_group = NULL;
            return -1;
        }
    }

    g_default_group->initialized = true;

    LOG_INFO("Distributed initialized: rank %d/%d, backend %s",
             rank, world_size,
             backend == DIST_BACKEND_NCCL ? "NCCL" :
             backend == DIST_BACKEND_MPI ? "MPI" : "Gloo");

    return 0;
}

DistProcessGroup* cml_dist_get_default_group(void) {
    return g_default_group;
}

int cml_dist_get_rank(void) {
    return g_default_group ? g_default_group->rank : 0;
}

int cml_dist_get_world_size(void) {
    return g_default_group ? g_default_group->world_size : 1;
}

bool cml_dist_is_initialized(void) {
    return g_default_group && g_default_group->initialized;
}

void cml_dist_destroy(void) {
    if (!g_default_group)
        return;

    if (g_default_group->ops) {
        if (g_default_group->ops->destroy)
            g_default_group->ops->destroy(g_default_group->backend_ctx);
        cml_dist_free_backend(g_default_group->ops);
    }

    free(g_default_group);
    g_default_group = NULL;

    LOG_INFO("Distributed destroyed");
}

int cml_dist_allreduce(Tensor* tensor, DistReduceOp op) {
    if (!g_default_group || !g_default_group->initialized) {
        LOG_ERROR("Distributed not initialized");
        return -1;
    }
    if (!g_default_group->ops->allreduce)
        return -1;

    return g_default_group->ops->allreduce(tensor, op, g_default_group->backend_ctx);
}

int cml_dist_broadcast(Tensor* tensor, int src_rank) {
    if (!g_default_group || !g_default_group->initialized)
        return -1;
    if (!g_default_group->ops->broadcast)
        return -1;

    return g_default_group->ops->broadcast(tensor, src_rank, g_default_group->backend_ctx);
}

int cml_dist_allgather(Tensor** output, Tensor* input) {
    if (!g_default_group || !g_default_group->initialized)
        return -1;
    if (!g_default_group->ops->allgather)
        return -1;

    return g_default_group->ops->allgather(output, input, g_default_group->backend_ctx);
}

int cml_dist_barrier(void) {
    if (!g_default_group || !g_default_group->initialized)
        return -1;
    if (!g_default_group->ops->barrier)
        return 0; /* No-op if no barrier */

    return g_default_group->ops->barrier(g_default_group->backend_ctx);
}

DistWork* cml_dist_allreduce_async(Tensor* tensor, DistReduceOp op) {
    if (!g_default_group || !g_default_group->initialized)
        return NULL;
    if (!g_default_group->ops->allreduce_async) {
        /* Fall back to sync allreduce */
        cml_dist_allreduce(tensor, op);
        DistWork* work = calloc(1, sizeof(DistWork));
        if (work) work->completed = true;
        return work;
    }

    return g_default_group->ops->allreduce_async(tensor, op, g_default_group->backend_ctx);
}

int cml_dist_wait(DistWork* work) {
    if (!work)
        return -1;
    if (work->completed)
        return 0;

    if (g_default_group && g_default_group->ops->wait)
        return g_default_group->ops->wait(work);

    return -1;
}

void cml_dist_work_free(DistWork* work) {
    if (!work)
        return;
    free(work->internal);
    free(work);
}
