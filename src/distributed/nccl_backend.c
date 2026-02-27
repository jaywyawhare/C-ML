/**
 * @file nccl_backend.c
 * @brief NCCL backend for distributed GPU training (via dlopen)
 */

#include "distributed/comm_backend.h"
#include "distributed/distributed.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef void* ncclComm_t;
typedef int ncclResult_t;
typedef int ncclRedOp_t;
typedef int ncclDataType_t;

typedef struct {
    void* handle;    /* dlopen handle */
    ncclResult_t (*ncclCommInitRank)(ncclComm_t*, int, void*, int);
    ncclResult_t (*ncclAllReduce)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, void*);
    ncclResult_t (*ncclBroadcast)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, void*);
    ncclResult_t (*ncclCommDestroy)(ncclComm_t);
    ncclResult_t (*ncclGetUniqueId)(void*);
    ncclComm_t comm;
} NCCLContext;

static int nccl_allreduce(Tensor* tensor, DistReduceOp op, void* ctx) {
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl || !nccl->ncclAllReduce || !tensor || !tensor->data)
        return -1;

    /* Map CML reduce op to NCCL op */
    int nccl_op = 0; /* ncclSum */
    switch (op) {
    case DIST_REDUCE_SUM: case DIST_REDUCE_AVG: nccl_op = 0; break;
    case DIST_REDUCE_PRODUCT: nccl_op = 1; break;
    case DIST_REDUCE_MAX: nccl_op = 2; break;
    case DIST_REDUCE_MIN: nccl_op = 3; break;
    }

    int result = nccl->ncclAllReduce(tensor->data, tensor->data, tensor->numel,
                                      0 /* ncclFloat32 */, nccl_op, nccl->comm,
                                      NULL /* default stream */);

    /* Average if requested */
    if (op == DIST_REDUCE_AVG && result == 0) {
        float scale = 1.0f / (float)cml_dist_get_world_size();
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < tensor->numel; i++)
            data[i] *= scale;
    }

    return result;
}

static int nccl_broadcast(Tensor* tensor, int src_rank, void* ctx) {
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl || !nccl->ncclBroadcast || !tensor)
        return -1;

    return nccl->ncclBroadcast(tensor->data, tensor->data, tensor->numel,
                                0 /* ncclFloat32 */, src_rank, nccl->comm,
                                NULL);
}

static int nccl_init(void* ctx, int world_size, int rank) {
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl)
        return -1;

    /* Get unique ID and init communicator */
    char unique_id[128];
    memset(unique_id, 0, sizeof(unique_id));

    if (nccl->ncclGetUniqueId) {
        nccl->ncclGetUniqueId(unique_id);
    }

    if (nccl->ncclCommInitRank) {
        return nccl->ncclCommInitRank(&nccl->comm, world_size, unique_id, rank);
    }

    return -1;
}

static void nccl_destroy(void* ctx) {
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl)
        return;

    if (nccl->comm && nccl->ncclCommDestroy)
        nccl->ncclCommDestroy(nccl->comm);

    if (nccl->handle)
        dlclose(nccl->handle);

    free(nccl);
}

DistCommOps* cml_dist_create_nccl_backend(void) {
    /* Try to load NCCL dynamically */
    void* handle = dlopen("libnccl.so", RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
    }
    if (!handle) {
        LOG_INFO("NCCL not found: %s", dlerror());
        return NULL;
    }

    NCCLContext* nccl = calloc(1, sizeof(NCCLContext));
    if (!nccl) {
        dlclose(handle);
        return NULL;
    }

    nccl->handle = handle;

    /* Load function pointers */
    *(void**)&nccl->ncclCommInitRank = dlsym(handle, "ncclCommInitRank");
    *(void**)&nccl->ncclAllReduce = dlsym(handle, "ncclAllReduce");
    *(void**)&nccl->ncclBroadcast = dlsym(handle, "ncclBroadcast");
    *(void**)&nccl->ncclCommDestroy = dlsym(handle, "ncclCommDestroy");
    *(void**)&nccl->ncclGetUniqueId = dlsym(handle, "ncclGetUniqueId");

    if (!nccl->ncclAllReduce) {
        LOG_WARNING("NCCL loaded but missing ncclAllReduce");
        dlclose(handle);
        free(nccl);
        return NULL;
    }

    DistCommOps* ops = calloc(1, sizeof(DistCommOps));
    if (!ops) {
        dlclose(handle);
        free(nccl);
        return NULL;
    }

    ops->allreduce = nccl_allreduce;
    ops->broadcast = nccl_broadcast;
    ops->init = nccl_init;
    ops->destroy = nccl_destroy;

    LOG_INFO("NCCL backend loaded successfully");
    return ops;
}
