#include "distributed/distributed.h"
#include "distributed/comm_backend.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include "alloc/cml_allocator.h"

static DistProcessGroup* g_default_group = NULL;
static pthread_mutex_t g_dist_mutex      = PTHREAD_MUTEX_INITIALIZER;

/** Create the default process group and bring up the requested backend, falling
 * back to Gloo if NCCL/MPI is unavailable. Negative @p world_size / @p rank are
 * auto-detected from WORLD_SIZE / RANK (or LOCAL_RANK). Idempotent; returns -1
 * on invalid rank/world or backend failure. Thread-safe. */
int cml_dist_init(DistBackendType backend, int world_size, int rank) {
    pthread_mutex_lock(&g_dist_mutex);

    if (g_default_group && g_default_group->initialized) {
        LOG_WARNING("Distributed already initialized");
        pthread_mutex_unlock(&g_dist_mutex);
        return 0;
    }

    /* Auto-detect from environment */
    if (world_size < 0) {
        const char* ws_env = getenv("WORLD_SIZE");
        world_size         = ws_env ? atoi(ws_env) : 1;
    }

    if (rank < 0) {
        const char* rank_env = getenv("RANK");
        if (!rank_env)
            rank_env = getenv("LOCAL_RANK");
        rank = rank_env ? atoi(rank_env) : 0;
    }

    /* Validate: an unset/garbage WORLD_SIZE (e.g. "0") would otherwise reach the
     * averaging path as 1/world_size = inf, and a rank outside [0,world_size)
     * corrupts every collective's peer indexing. */
    if (world_size < 1) {
        LOG_ERROR("Distributed init: invalid world_size=%d (must be >= 1)", world_size);
        pthread_mutex_unlock(&g_dist_mutex);
        return -1;
    }
    if (rank < 0 || rank >= world_size) {
        LOG_ERROR("Distributed init: invalid rank=%d for world_size=%d", rank, world_size);
        pthread_mutex_unlock(&g_dist_mutex);
        return -1;
    }

    g_default_group = cml_calloc(1, sizeof(DistProcessGroup));
    if (!g_default_group) {
        pthread_mutex_unlock(&g_dist_mutex);
        return -1;
    }

    g_default_group->rank       = rank;
    g_default_group->world_size = world_size;
    g_default_group->backend    = backend;

    /* Create backend ops */
    DistCommOps* ops = NULL;
    switch (backend) {
    case DIST_BACKEND_NCCL:
        ops = cml_dist_create_nccl_backend();
        if (!ops) {
            LOG_WARNING("NCCL unavailable, falling back to Gloo");
            ops                      = cml_dist_create_gloo_backend();
            g_default_group->backend = DIST_BACKEND_GLOO;
        }
        break;
    case DIST_BACKEND_MPI:
        ops = cml_dist_create_mpi_backend();
        if (!ops) {
            LOG_WARNING("MPI unavailable, falling back to Gloo");
            ops                      = cml_dist_create_gloo_backend();
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
        cml_free(g_default_group);
        g_default_group = NULL;
        pthread_mutex_unlock(&g_dist_mutex);
        return -1;
    }

    g_default_group->ops = ops;

    /* Store backend context on the process group */
    g_default_group->backend_ctx = ops->backend_ctx;

    /* Initialize backend */
    if (ops->init) {
        int result = ops->init(ops->backend_ctx, world_size, rank);
        if (result != 0) {
            LOG_ERROR("Backend initialization failed");
            cml_dist_free_backend(ops, g_default_group->backend_ctx);
            cml_free(g_default_group);
            g_default_group = NULL;
            pthread_mutex_unlock(&g_dist_mutex);
            return -1;
        }
    }

    g_default_group->initialized = true;

    LOG_INFO("Distributed initialized: rank %d/%d, backend %s", rank, world_size,
             backend == DIST_BACKEND_NCCL  ? "NCCL"
             : backend == DIST_BACKEND_MPI ? "MPI"
                                           : "Gloo");

    pthread_mutex_unlock(&g_dist_mutex);
    return 0;
}

/** The default process group, or NULL if cml_dist_init has not run. */
DistProcessGroup* cml_dist_get_default_group(void) { return g_default_group; }

/** This process's rank, or 0 when distributed is uninitialized. */
int cml_dist_get_rank(void) { return g_default_group ? g_default_group->rank : 0; }

/** The group's world size, or 1 when distributed is uninitialized. */
int cml_dist_get_world_size(void) { return g_default_group ? g_default_group->world_size : 1; }

/** True once a default group exists and its backend has been initialized. */
bool cml_dist_is_initialized(void) { return g_default_group && g_default_group->initialized; }

/** Tear down the backend and release the default process group. Thread-safe and
 * a no-op if nothing was initialized. */
void cml_dist_destroy(void) {
    pthread_mutex_lock(&g_dist_mutex);

    if (!g_default_group) {
        pthread_mutex_unlock(&g_dist_mutex);
        return;
    }

    if (g_default_group->ops)
        cml_dist_free_backend(g_default_group->ops, g_default_group->backend_ctx);

    cml_free(g_default_group);
    g_default_group = NULL;

    pthread_mutex_unlock(&g_dist_mutex);
    LOG_INFO("Distributed destroyed");
}

/** Reduce @p tensor in place across all ranks with @p op. Dispatches to the
 * active backend; returns -1 if uninitialized or the op is unsupported. */
int cml_dist_allreduce(Tensor* tensor, DistReduceOp op) {
    if (!g_default_group || !g_default_group->initialized) {
        LOG_ERROR("Distributed not initialized");
        return -1;
    }
    if (!g_default_group->ops->allreduce)
        return -1;

    return g_default_group->ops->allreduce(tensor, op, g_default_group->backend_ctx);
}

/** Broadcast @p tensor from @p src_rank to every rank in place. Returns -1 if
 * uninitialized or unsupported by the backend. */
int cml_dist_broadcast(Tensor* tensor, int src_rank) {
    if (!g_default_group || !g_default_group->initialized)
        return -1;
    if (!g_default_group->ops->broadcast)
        return -1;

    return g_default_group->ops->broadcast(tensor, src_rank, g_default_group->backend_ctx);
}

/** Gather each rank's @p input into the per-rank @p output array on all ranks.
 * Returns -1 if uninitialized or unsupported by the backend. */
int cml_dist_allgather(Tensor** output, Tensor* input) {
    if (!g_default_group || !g_default_group->initialized)
        return -1;
    if (!g_default_group->ops->allgather)
        return -1;

    return g_default_group->ops->allgather(output, input, g_default_group->backend_ctx);
}

/** Block until all ranks reach the barrier. Returns 0 (treated as a no-op) if
 * the backend has no barrier op, -1 if uninitialized. */
int cml_dist_barrier(void) {
    if (!g_default_group || !g_default_group->initialized)
        return -1;
    if (!g_default_group->ops->barrier)
        return 0; /* No-op if no barrier */

    return g_default_group->ops->barrier(g_default_group->backend_ctx);
}

/** Point-to-point send of @p tensor to @p dst_rank with message @p tag. Returns
 * -1 if uninitialized or the backend lacks send. */
int cml_dist_send(Tensor* tensor, int dst_rank, int tag) {
    if (!g_default_group || !g_default_group->initialized)
        return -1;
    if (!g_default_group->ops->send)
        return -1;
    return g_default_group->ops->send(tensor, dst_rank, tag, g_default_group->backend_ctx);
}

/** Point-to-point receive into @p tensor from @p src_rank matching @p tag.
 * Returns -1 if uninitialized or the backend lacks recv. */
int cml_dist_recv(Tensor* tensor, int src_rank, int tag) {
    if (!g_default_group || !g_default_group->initialized)
        return -1;
    if (!g_default_group->ops->recv)
        return -1;
    return g_default_group->ops->recv(tensor, src_rank, tag, g_default_group->backend_ctx);
}

/** Start a non-blocking all-reduce, returning a DistWork handle to wait on. If
 * the backend has no async path, runs synchronously and returns an
 * already-completed handle. NULL on uninitialized or allocation failure. */
DistWork* cml_dist_allreduce_async(Tensor* tensor, DistReduceOp op) {
    if (!g_default_group || !g_default_group->initialized)
        return NULL;
    if (!g_default_group->ops->allreduce_async) {
        /* Fall back to sync allreduce */
        int rc         = cml_dist_allreduce(tensor, op);
        DistWork* work = cml_calloc(1, sizeof(DistWork));
        if (work) {
            work->completed  = true;
            work->error_code = rc;
        }
        return work;
    }

    return g_default_group->ops->allreduce_async(tensor, op, g_default_group->backend_ctx);
}

/** Block until @p work finishes and return its error code; returns immediately
 * for an already-completed handle. */
int cml_dist_wait(DistWork* work) {
    if (!work)
        return -1;
    if (work->completed)
        return work->error_code;

    if (g_default_group && g_default_group->ops->wait)
        return g_default_group->ops->wait(work);

    return -1;
}

/** Release a DistWork handle and its backend-internal state. */
void cml_dist_work_free(DistWork* work) {
    if (!work)
        return;
    cml_free(work->internal);
    cml_free(work);
}
