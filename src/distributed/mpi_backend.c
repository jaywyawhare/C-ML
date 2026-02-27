/**
 * @file mpi_backend.c
 * @brief MPI backend for distributed training (via dlopen)
 */

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
    bool initialized;
} MPIContext;

static int mpi_allreduce(Tensor* tensor, DistReduceOp op, void* ctx) {
    MPIContext* mpi = (MPIContext*)ctx;
    if (!mpi || !mpi->MPI_Allreduce || !tensor || !tensor->data)
        return -1;

    int mpi_op;
    switch (op) {
    case DIST_REDUCE_SUM: case DIST_REDUCE_AVG: mpi_op = CML_MPI_SUM; break;
    case DIST_REDUCE_PRODUCT: mpi_op = CML_MPI_PROD; break;
    case DIST_REDUCE_MAX: mpi_op = CML_MPI_MAX; break;
    case DIST_REDUCE_MIN: mpi_op = CML_MPI_MIN; break;
    }

    /* MPI_Allreduce with MPI_IN_PLACE */
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
    MPIContext* mpi = (MPIContext*)ctx;
    if (!mpi || !mpi->MPI_Bcast || !tensor || !tensor->data)
        return -1;

    return mpi->MPI_Bcast(tensor->data, (int)tensor->numel, CML_MPI_FLOAT,
                           src_rank, CML_MPI_COMM_WORLD);
}

static int mpi_barrier(void* ctx) {
    MPIContext* mpi = (MPIContext*)ctx;
    if (!mpi || !mpi->MPI_Barrier)
        return -1;

    return mpi->MPI_Barrier(CML_MPI_COMM_WORLD);
}

static int mpi_init(void* ctx, int world_size, int rank) {
    MPIContext* mpi = (MPIContext*)ctx;
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
    MPIContext* mpi = (MPIContext*)ctx;
    if (!mpi)
        return;

    if (mpi->MPI_Finalize && mpi->initialized)
        mpi->MPI_Finalize();

    if (mpi->handle)
        dlclose(mpi->handle);

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

    if (!mpi->MPI_Allreduce) {
        LOG_WARNING("MPI loaded but missing MPI_Allreduce");
        dlclose(handle);
        free(mpi);
        return NULL;
    }

    DistCommOps* ops = calloc(1, sizeof(DistCommOps));
    if (!ops) {
        dlclose(handle);
        free(mpi);
        return NULL;
    }

    ops->allreduce = mpi_allreduce;
    ops->broadcast = mpi_broadcast;
    ops->barrier = mpi_barrier;
    ops->init = mpi_init;
    ops->destroy = mpi_destroy;

    LOG_INFO("MPI backend loaded successfully");
    return ops;
}
