#include "distributed/comm_backend.h"
#include "distributed/distributed.h"
#include "core/logging.h"
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>

/* MPI constants */
#define CML_MPI_COMM_WORLD ((void*)0x44000000)
#define CML_MPI_FLOAT 0x4c00040a
#define CML_MPI_SUM 0x58000003
#define CML_MPI_PROD 0x58000004
#define CML_MPI_MAX 0x58000001
#define CML_MPI_MIN 0x58000002

typedef struct {
    void* handle;
    int (*MPI_Init)(int*, char***);
    int (*MPI_Finalize)(void);
    int (*MPI_Comm_rank)(void*, int*);
    int (*MPI_Comm_size)(void*, int*);
    int (*MPI_Allreduce)(const void*, void*, int, int, int, void*);
    int (*MPI_Bcast)(void*, int, int, int, void*);
    int (*MPI_Barrier)(void*);
    int (*MPI_Send)(const void*, int, int, int, int, void*);
    int (*MPI_Recv)(void*, int, int, int, int, void*, void*);
    int (*MPI_Allgather)(const void*, int, int, void*, int, int, void*);
    int (*MPI_Reduce_scatter)(const void*, void*, const int*, int, int, void*);
    int (*MPI_Iallreduce)(const void*, void*, int, int, int, void*, void*);
    int (*MPI_Wait)(void*, void*);
    bool initialized;
} MPIContext;

static MPIContext* g_mpi_ctx = NULL;

static MPIContext* mpi_get_ctx(void* ctx) {
    if (ctx)
        return (MPIContext*)ctx;
    return g_mpi_ctx;
}

static int mpi_op_to_const(DistReduceOp op) {
    switch (op) {
    case DIST_REDUCE_SUM: case DIST_REDUCE_AVG: return CML_MPI_SUM;
    case DIST_REDUCE_PRODUCT: return CML_MPI_PROD;
    case DIST_REDUCE_MAX: return CML_MPI_MAX;
    case DIST_REDUCE_MIN: return CML_MPI_MIN;
    }
    return CML_MPI_SUM;
}

static int mpi_allreduce(Tensor* tensor, DistReduceOp op, void* ctx) {
    MPIContext* mpi = mpi_get_ctx(ctx);
    if (!mpi || !mpi->MPI_Allreduce || !tensor || !tensor->data)
        return -1;

    int mpi_op = mpi_op_to_const(op);

    float* sendbuf = (float*)tensor->data;
    float* recvbuf = (float*)malloc(tensor->numel * sizeof(float));
    if (!recvbuf)
        return -1;

    int result = mpi->MPI_Allreduce(sendbuf, recvbuf, (int)tensor->numel,
                                     CML_MPI_FLOAT, mpi_op, CML_MPI_COMM_WORLD);

    if (result == 0)
        memcpy(tensor->data, recvbuf, tensor->numel * sizeof(float));

    free(recvbuf);

    if (op == DIST_REDUCE_AVG && result == 0) {
        float scale = 1.0f / (float)cml_dist_get_world_size();
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < tensor->numel; i++)
            data[i] *= scale;
    }

    return result;
}

static int mpi_broadcast(Tensor* tensor, int src_rank, void* ctx) {
    MPIContext* mpi = mpi_get_ctx(ctx);
    if (!mpi || !mpi->MPI_Bcast || !tensor || !tensor->data)
        return -1;

    return mpi->MPI_Bcast(tensor->data, (int)tensor->numel, CML_MPI_FLOAT,
                           src_rank, CML_MPI_COMM_WORLD);
}

static int mpi_barrier(void* ctx) {
    MPIContext* mpi = mpi_get_ctx(ctx);
    if (!mpi || !mpi->MPI_Barrier)
        return -1;

    return mpi->MPI_Barrier(CML_MPI_COMM_WORLD);
}

static int mpi_allgather(Tensor** output, Tensor* input, void* ctx) {
    MPIContext* mpi = mpi_get_ctx(ctx);
    if (!mpi || !mpi->MPI_Allgather || !output || !input || !input->data)
        return -1;

    int world_size = cml_dist_get_world_size();
    int sendcount = (int)input->numel;

    float* recvbuf = (float*)malloc((size_t)world_size * sendcount * sizeof(float));
    if (!recvbuf)
        return -1;

    int result = mpi->MPI_Allgather(input->data, sendcount, CML_MPI_FLOAT,
                                     recvbuf, sendcount, CML_MPI_FLOAT,
                                     CML_MPI_COMM_WORLD);

    if (result == 0) {
        for (int i = 0; i < world_size; i++) {
            if (output[i] && output[i]->data) {
                memcpy(output[i]->data, recvbuf + i * sendcount,
                       sendcount * sizeof(float));
            }
        }
    }

    free(recvbuf);
    return result;
}

static int mpi_reduce_scatter(Tensor* output, Tensor* input, DistReduceOp op, void* ctx) {
    MPIContext* mpi = mpi_get_ctx(ctx);
    if (!mpi || !mpi->MPI_Reduce_scatter || !output || !input ||
        !output->data || !input->data)
        return -1;

    int world_size = cml_dist_get_world_size();
    int chunk_size = (int)(input->numel / (size_t)world_size);
    int mpi_op = mpi_op_to_const(op);

    int* recvcounts = (int*)malloc((size_t)world_size * sizeof(int));
    if (!recvcounts)
        return -1;

    for (int i = 0; i < world_size; i++)
        recvcounts[i] = chunk_size;

    int result = mpi->MPI_Reduce_scatter(input->data, output->data,
                                          recvcounts, CML_MPI_FLOAT,
                                          mpi_op, CML_MPI_COMM_WORLD);

    free(recvcounts);

    if (op == DIST_REDUCE_AVG && result == 0) {
        float scale = 1.0f / (float)world_size;
        float* data = (float*)output->data;
        for (size_t i = 0; i < output->numel; i++)
            data[i] *= scale;
    }

    return result;
}

static int mpi_send(Tensor* tensor, int dst_rank, int tag, void* ctx) {
    MPIContext* mpi = mpi_get_ctx(ctx);
    if (!mpi || !mpi->MPI_Send || !tensor || !tensor->data)
        return -1;

    return mpi->MPI_Send(tensor->data, (int)tensor->numel, CML_MPI_FLOAT,
                          dst_rank, tag, CML_MPI_COMM_WORLD);
}

static int mpi_recv(Tensor* tensor, int src_rank, int tag, void* ctx) {
    MPIContext* mpi = mpi_get_ctx(ctx);
    if (!mpi || !mpi->MPI_Recv || !tensor || !tensor->data)
        return -1;

    return mpi->MPI_Recv(tensor->data, (int)tensor->numel, CML_MPI_FLOAT,
                          src_rank, tag, CML_MPI_COMM_WORLD, NULL);
}

static DistWork* mpi_allreduce_async(Tensor* tensor, DistReduceOp op, void* ctx) {
    MPIContext* mpi = mpi_get_ctx(ctx);
    if (!mpi || !tensor || !tensor->data)
        return NULL;

    DistWork* work = calloc(1, sizeof(DistWork));
    if (!work)
        return NULL;

    int mpi_op = mpi_op_to_const(op);

    if (mpi->MPI_Iallreduce) {
        float* recvbuf = (float*)malloc(tensor->numel * sizeof(float));
        if (!recvbuf) {
            free(work);
            return NULL;
        }

        /* Allocate MPI_Request handle (opaque, typically pointer-sized) */
        void* request = calloc(1, 128);
        if (!request) {
            free(recvbuf);
            free(work);
            return NULL;
        }

        int result = mpi->MPI_Iallreduce(tensor->data, recvbuf,
                                           (int)tensor->numel, CML_MPI_FLOAT,
                                           mpi_op, CML_MPI_COMM_WORLD, request);

        if (result != 0) {
            free(request);
            free(recvbuf);
            free(work);
            return NULL;
        }

        /* Store request as internal handle; recvbuf will be copied on wait */
        work->internal = request;
        work->completed = false;
        work->error_code = 0;

        /* We need to copy recvbuf back after wait. Store both in a wrapper. */
        /* For simplicity, just do the copy after MPI_Wait in mpi_wait. */
        /* We lose the recvbuf pointer here - use sync fallback for correctness. */
        /* Actually, store recvbuf alongside the request. */
        /* Pack: [request_data(128 bytes)][recvbuf_ptr(8 bytes)][tensor_ptr(8 bytes)][numel(8 bytes)] */
        memcpy((char*)request + 64, &recvbuf, sizeof(float*));
        memcpy((char*)request + 64 + sizeof(float*), &tensor, sizeof(Tensor*));
        memcpy((char*)request + 64 + sizeof(float*) + sizeof(Tensor*), &tensor->numel, sizeof(size_t));

        if (op == DIST_REDUCE_AVG) {
            int ws = cml_dist_get_world_size();
            memcpy((char*)request + 64 + sizeof(float*) + sizeof(Tensor*) + sizeof(size_t), &ws, sizeof(int));
        }

    } else {
        /* Fallback to synchronous allreduce */
        int result = mpi_allreduce(tensor, op, ctx);
        work->completed = true;
        work->error_code = result;
    }

    return work;
}

static int mpi_wait(DistWork* work) {
    if (!work)
        return -1;
    if (work->completed)
        return work->error_code;

    MPIContext* mpi = g_mpi_ctx;
    if (!mpi || !mpi->MPI_Wait || !work->internal)
        return -1;

    void* request = work->internal;
    int result = mpi->MPI_Wait(request, NULL);

    if (result == 0) {
        /* Retrieve packed recvbuf, tensor, and numel */
        float* recvbuf = NULL;
        Tensor* tensor = NULL;
        size_t numel = 0;
        memcpy(&recvbuf, (char*)request + 64, sizeof(float*));
        memcpy(&tensor, (char*)request + 64 + sizeof(float*), sizeof(Tensor*));
        memcpy(&numel, (char*)request + 64 + sizeof(float*) + sizeof(Tensor*), sizeof(size_t));

        if (recvbuf && tensor && tensor->data && numel > 0)
            memcpy(tensor->data, recvbuf, numel * sizeof(float));

        free(recvbuf);
    }

    work->completed = true;
    work->error_code = result;

    return result;
}

static int mpi_init(void* ctx, int world_size, int rank) {
    MPIContext* mpi = mpi_get_ctx(ctx);
    (void)world_size;
    (void)rank;
    if (!mpi)
        return -1;

    if (mpi->MPI_Init && !mpi->initialized) {
        int result = mpi->MPI_Init(NULL, NULL);
        if (result != 0)
            return -1;
        mpi->initialized = true;
    }

    return 0;
}

static void mpi_destroy(void* ctx) {
    MPIContext* mpi = mpi_get_ctx(ctx);
    if (!mpi)
        return;

    if (mpi->MPI_Finalize && mpi->initialized)
        mpi->MPI_Finalize();

    if (mpi->handle)
        dlclose(mpi->handle);

    if (g_mpi_ctx == mpi)
        g_mpi_ctx = NULL;

    free(mpi);
}

DistCommOps* cml_dist_create_mpi_backend(void) {
    void* handle = dlopen("libmpi.so", RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        handle = dlopen("libmpi.so.40", RTLD_NOW | RTLD_LOCAL);
    }
    if (!handle) {
        handle = dlopen("libmpi.so.12", RTLD_NOW | RTLD_LOCAL);
    }
    if (!handle) {
        LOG_INFO("MPI not found: %s", dlerror());
        return NULL;
    }

    MPIContext* mpi = calloc(1, sizeof(MPIContext));
    if (!mpi) {
        dlclose(handle);
        return NULL;
    }

    mpi->handle = handle;

    *(void**)&mpi->MPI_Init = dlsym(handle, "MPI_Init");
    *(void**)&mpi->MPI_Finalize = dlsym(handle, "MPI_Finalize");
    *(void**)&mpi->MPI_Comm_rank = dlsym(handle, "MPI_Comm_rank");
    *(void**)&mpi->MPI_Comm_size = dlsym(handle, "MPI_Comm_size");
    *(void**)&mpi->MPI_Allreduce = dlsym(handle, "MPI_Allreduce");
    *(void**)&mpi->MPI_Bcast = dlsym(handle, "MPI_Bcast");
    *(void**)&mpi->MPI_Barrier = dlsym(handle, "MPI_Barrier");
    *(void**)&mpi->MPI_Send = dlsym(handle, "MPI_Send");
    *(void**)&mpi->MPI_Recv = dlsym(handle, "MPI_Recv");
    *(void**)&mpi->MPI_Allgather = dlsym(handle, "MPI_Allgather");
    *(void**)&mpi->MPI_Reduce_scatter = dlsym(handle, "MPI_Reduce_scatter");
    *(void**)&mpi->MPI_Iallreduce = dlsym(handle, "MPI_Iallreduce");
    *(void**)&mpi->MPI_Wait = dlsym(handle, "MPI_Wait");

    if (!mpi->MPI_Allreduce) {
        LOG_WARNING("MPI loaded but missing MPI_Allreduce");
        dlclose(handle);
        free(mpi);
        return NULL;
    }

    g_mpi_ctx = mpi;

    DistCommOps* ops = calloc(1, sizeof(DistCommOps));
    if (!ops) {
        g_mpi_ctx = NULL;
        dlclose(handle);
        free(mpi);
        return NULL;
    }

    ops->allreduce = mpi_allreduce;
    ops->broadcast = mpi_broadcast;
    ops->allgather = mpi_allgather;
    ops->reduce_scatter = mpi_reduce_scatter;
    ops->barrier = mpi_barrier;
    ops->send = mpi_send;
    ops->recv = mpi_recv;
    ops->allreduce_async = mpi_allreduce_async;
    ops->wait = mpi_wait;
    ops->init = mpi_init;
    ops->destroy = mpi_destroy;
    ops->backend_ctx = mpi;

    LOG_INFO("MPI backend loaded successfully");
    return ops;
}
