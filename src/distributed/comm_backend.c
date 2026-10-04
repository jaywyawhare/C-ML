#include "distributed/comm_backend.h"
#include "distributed/distributed.h"
#include "core/logging.h"
#include <stdlib.h>
#include "alloc/cml_allocator.h"

/** Tear down a backend: run its destroy hook on @p backend_ctx (if any), then
 * free the ops table itself. Safe to call with a NULL ops. */
void cml_dist_free_backend(DistCommOps* ops, void* backend_ctx) {
    if (!ops)
        return;

    if (ops->destroy)
        ops->destroy(backend_ctx);

    cml_free(ops);
}

/** Probe backends in descending preference (NCCL, then MPI, then Gloo) and
 * return the first that is usable, or NULL if none can be created. */
DistCommOps* cml_dist_auto_select_backend(void) {
    DistCommOps* ops = NULL;

    /* Try NCCL first (best GPU performance) */
    ops = cml_dist_create_nccl_backend();
    if (ops) {
        LOG_INFO("Auto-selected NCCL backend");
        return ops;
    }

    /* Try MPI (widely available, CPU/GPU) */
    ops = cml_dist_create_mpi_backend();
    if (ops) {
        LOG_INFO("Auto-selected MPI backend");
        return ops;
    }

    /* Fall back to Gloo (pure CPU, always available) */
    ops = cml_dist_create_gloo_backend();
    if (ops) {
        LOG_INFO("Auto-selected Gloo backend");
        return ops;
    }

    LOG_ERROR("No distributed backend available");
    return NULL;
}

/** Human-readable name for a backend type; "Unknown" for unrecognized values. */
const char* cml_dist_backend_name(DistBackendType type) {
    switch (type) {
    case DIST_BACKEND_NCCL:
        return "NCCL";
    case DIST_BACKEND_MPI:
        return "MPI";
    case DIST_BACKEND_GLOO:
        return "Gloo";
    default:
        return "Unknown";
    }
}
