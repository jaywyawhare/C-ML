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
    ncclResult_t (*ncclAllGather)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, void*);
    ncclResult_t (*ncclReduceScatter)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, void*);
    ncclResult_t (*ncclSend)(const void*, size_t, ncclDataType_t, int, ncclComm_t, void*);
    ncclResult_t (*ncclRecv)(void*, size_t, ncclDataType_t, int, ncclComm_t, void*);
    ncclResult_t (*ncclGroupStart)(void);
    ncclResult_t (*ncclGroupEnd)(void);
    ncclResult_t (*ncclCommGetAsyncError)(ncclComm_t, ncclResult_t*);
    ncclComm_t comm;
    void* stream;    /* CUDA stream (NULL = default stream) */
} NCCLContext;

/* Static context pointer (same pattern as MPI backend) */
static NCCLContext* g_nccl_ctx = NULL;

static int nccl_map_reduce_op(DistReduceOp op) {
    switch (op) {
    case DIST_REDUCE_SUM: case DIST_REDUCE_AVG: return 0; /* ncclSum */
    case DIST_REDUCE_PRODUCT: return 1; /* ncclProd */
    case DIST_REDUCE_MAX: return 2;     /* ncclMax */
    case DIST_REDUCE_MIN: return 3;     /* ncclMin */
    }
    return 0;
}

static int nccl_allreduce(Tensor* tensor, DistReduceOp op, void* ctx) {
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl) nccl = g_nccl_ctx;
    if (!nccl || !nccl->ncclAllReduce || !tensor || !tensor->data)
        return -1;

    int nccl_op = nccl_map_reduce_op(op);

    int result = nccl->ncclAllReduce(tensor->data, tensor->data, tensor->numel,
                                      0 /* ncclFloat32 */, nccl_op, nccl->comm,
                                      nccl->stream);

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
    if (!nccl) nccl = g_nccl_ctx;
    if (!nccl || !nccl->ncclBroadcast || !tensor)
        return -1;

    return nccl->ncclBroadcast(tensor->data, tensor->data, tensor->numel,
                                0 /* ncclFloat32 */, src_rank, nccl->comm,
                                nccl->stream);
}

static int nccl_allgather(Tensor** output, Tensor* input, void* ctx) {
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl) nccl = g_nccl_ctx;
    if (!nccl || !nccl->ncclAllGather || !output || !input || !input->data)
        return -1;

    int world_size = cml_dist_get_world_size();
    if (world_size <= 0)
        return -1;

    /* Allocate flat gather buffer: world_size * input->numel elements */
    size_t chunk_size = input->numel;
    size_t total_size = chunk_size * (size_t)world_size;
    float* gather_buf = (float*)malloc(total_size * sizeof(float));
    if (!gather_buf)
        return -1;

    /* ncclAllGather: each rank sends chunk_size elements, receives world_size * chunk_size */
    int result = nccl->ncclAllGather(input->data, gather_buf, chunk_size,
                                      0 /* ncclFloat32 */, nccl->comm,
                                      nccl->stream);

    if (result != 0) {
        free(gather_buf);
        return result;
    }

    /* Distribute gathered data into output tensors */
    for (int i = 0; i < world_size; i++) {
        if (output[i] && output[i]->data) {
            size_t copy_size = output[i]->numel < chunk_size ? output[i]->numel : chunk_size;
            memcpy(output[i]->data, gather_buf + i * chunk_size, copy_size * sizeof(float));
        }
    }

    free(gather_buf);
    return 0;
}

static int nccl_reduce_scatter(Tensor* output, Tensor* input, DistReduceOp op, void* ctx) {
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl) nccl = g_nccl_ctx;
    if (!nccl || !nccl->ncclReduceScatter || !output || !input || !input->data || !output->data)
        return -1;

    int nccl_op = nccl_map_reduce_op(op);

    int result = nccl->ncclReduceScatter(input->data, output->data, output->numel,
                                          0 /* ncclFloat32 */, nccl_op, nccl->comm,
                                          nccl->stream);

    /* Average if requested */
    if (op == DIST_REDUCE_AVG && result == 0) {
        float scale = 1.0f / (float)cml_dist_get_world_size();
        float* data = (float*)output->data;
        for (size_t i = 0; i < output->numel; i++)
            data[i] *= scale;
    }

    return result;
}

static int nccl_barrier(void* ctx) {
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl) nccl = g_nccl_ctx;
    if (!nccl || !nccl->ncclAllReduce)
        return -1;

    /* Standard NCCL barrier pattern: allreduce a single dummy element */
    float dummy = 0.0f;
    return nccl->ncclAllReduce(&dummy, &dummy, 1,
                                0 /* ncclFloat32 */, 0 /* ncclSum */,
                                nccl->comm, nccl->stream);
}

static int nccl_send(Tensor* tensor, int dst_rank, int tag, void* ctx) {
    (void)tag; /* NCCL does not support message tags */
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl) nccl = g_nccl_ctx;
    if (!nccl || !nccl->ncclSend || !nccl->ncclGroupStart || !nccl->ncclGroupEnd)
        return -1;
    if (!tensor || !tensor->data)
        return -1;

    int result;
    result = nccl->ncclGroupStart();
    if (result != 0)
        return result;

    result = nccl->ncclSend(tensor->data, tensor->numel,
                             0 /* ncclFloat32 */, dst_rank,
                             nccl->comm, nccl->stream);
    if (result != 0)
        return result;

    return nccl->ncclGroupEnd();
}

static int nccl_recv(Tensor* tensor, int src_rank, int tag, void* ctx) {
    (void)tag; /* NCCL does not support message tags */
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl) nccl = g_nccl_ctx;
    if (!nccl || !nccl->ncclRecv || !nccl->ncclGroupStart || !nccl->ncclGroupEnd)
        return -1;
    if (!tensor || !tensor->data)
        return -1;

    int result;
    result = nccl->ncclGroupStart();
    if (result != 0)
        return result;

    result = nccl->ncclRecv(tensor->data, tensor->numel,
                             0 /* ncclFloat32 */, src_rank,
                             nccl->comm, nccl->stream);
    if (result != 0)
        return result;

    return nccl->ncclGroupEnd();
}

static DistWork* nccl_allreduce_async(Tensor* tensor, DistReduceOp op, void* ctx) {
    NCCLContext* nccl = (NCCLContext*)ctx;
    if (!nccl) nccl = g_nccl_ctx;
    if (!nccl || !nccl->ncclAllReduce || !tensor || !tensor->data)
        return NULL;

    int nccl_op = nccl_map_reduce_op(op);

    /* Launch the allreduce without synchronizing */
    int result = nccl->ncclAllReduce(tensor->data, tensor->data, tensor->numel,
                                      0 /* ncclFloat32 */, nccl_op, nccl->comm,
                                      nccl->stream);

    DistWork* work = (DistWork*)calloc(1, sizeof(DistWork));
    if (!work)
        return NULL;

    if (result != 0) {
        work->completed = true;
        work->error_code = result;
        return work;
    }

    /*
     * If we had CUDA event support, we would record an event here
     * and store it in work->internal. Without CUDA runtime linkage,
     * mark as completed immediately (the operation was enqueued on
     * the NCCL stream and will complete in stream order).
     */
    work->internal = NULL;
    work->completed = true;
    work->error_code = 0;

    /* Handle AVG post-scaling */
    if (op == DIST_REDUCE_AVG) {
        float scale = 1.0f / (float)cml_dist_get_world_size();
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < tensor->numel; i++)
            data[i] *= scale;
    }

    return work;
}

static int nccl_wait(DistWork* work) {
    if (!work)
        return -1;

    if (work->completed)
        return work->error_code;

    /*
     * If work->internal held a CUDA event, we would synchronize on
     * it here (e.g., cudaEventSynchronize). Since we currently mark
     * operations as completed immediately, just return.
     */
    work->completed = true;
    return work->error_code;
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
        int result = nccl->ncclCommInitRank(&nccl->comm, world_size, unique_id, rank);
        if (result == 0)
            g_nccl_ctx = nccl;
        return result;
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

    if (g_nccl_ctx == nccl)
        g_nccl_ctx = NULL;

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
    nccl->stream = NULL; /* default CUDA stream */

    /* Load function pointers */
    *(void**)&nccl->ncclCommInitRank = dlsym(handle, "ncclCommInitRank");
    *(void**)&nccl->ncclAllReduce = dlsym(handle, "ncclAllReduce");
    *(void**)&nccl->ncclBroadcast = dlsym(handle, "ncclBroadcast");
    *(void**)&nccl->ncclCommDestroy = dlsym(handle, "ncclCommDestroy");
    *(void**)&nccl->ncclGetUniqueId = dlsym(handle, "ncclGetUniqueId");
    *(void**)&nccl->ncclAllGather = dlsym(handle, "ncclAllGather");
    *(void**)&nccl->ncclReduceScatter = dlsym(handle, "ncclReduceScatter");
    *(void**)&nccl->ncclSend = dlsym(handle, "ncclSend");
    *(void**)&nccl->ncclRecv = dlsym(handle, "ncclRecv");
    *(void**)&nccl->ncclGroupStart = dlsym(handle, "ncclGroupStart");
    *(void**)&nccl->ncclGroupEnd = dlsym(handle, "ncclGroupEnd");
    *(void**)&nccl->ncclCommGetAsyncError = dlsym(handle, "ncclCommGetAsyncError");

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
    ops->allgather = nccl_allgather;
    ops->reduce_scatter = nccl_reduce_scatter;
    ops->barrier = nccl_barrier;
    ops->send = nccl_send;
    ops->recv = nccl_recv;
    ops->allreduce_async = nccl_allreduce_async;
    ops->wait = nccl_wait;
    ops->init = nccl_init;
    ops->destroy = nccl_destroy;
    ops->backend_ctx = nccl;

    LOG_INFO("NCCL backend loaded successfully");
    return ops;
}
