#include "distributed/ring_allreduce.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void apply_reduce_op(float* dst, const float* src, size_t n, DistReduceOp op) {
    for (size_t i = 0; i < n; i++) {
        switch (op) {
        case DIST_REDUCE_SUM:
        case DIST_REDUCE_AVG:
            dst[i] += src[i];
            break;
        case DIST_REDUCE_PRODUCT:
            dst[i] *= src[i];
            break;
        case DIST_REDUCE_MAX:
            if (src[i] > dst[i]) dst[i] = src[i];
            break;
        case DIST_REDUCE_MIN:
            if (src[i] < dst[i]) dst[i] = src[i];
            break;
        }
    }
}

int cml_ring_allreduce(float* data, size_t count, int world_size, int rank,
                       DistReduceOp op, DistCommOps* ops, void* ctx) {
    if (!data || !ops || world_size <= 0 || rank < 0 || rank >= world_size)
        return -1;

    /* Trivial case: single process */
    if (world_size == 1) {
        if (op == DIST_REDUCE_AVG) {
            /* Average of 1 process is identity */
        }
        return 0;
    }

    /* Need send/recv for ring communication */
    if (!ops->send || !ops->recv)
        return -1;

    size_t chunk_size = (count + (size_t)world_size - 1) / (size_t)world_size;
    float* recv_buf = (float*)malloc(chunk_size * sizeof(float));
    if (!recv_buf) return -1;

    int left  = (rank - 1 + world_size) % world_size;
    int right = (rank + 1) % world_size;

    /* Create temporary tensors for send/recv */
    int send_shape[1], recv_shape[1];

    /* Phase 1: Reduce-scatter ring */
    for (int step = 0; step < world_size - 1; step++) {
        int send_chunk = (rank - step + world_size) % world_size;
        int recv_chunk = (rank - step - 1 + world_size) % world_size;

        size_t send_offset = (size_t)send_chunk * chunk_size;
        size_t recv_offset = (size_t)recv_chunk * chunk_size;
        size_t send_count  = chunk_size;
        size_t recv_count  = chunk_size;

        /* Clamp to actual data size */
        if (send_offset + send_count > count)
            send_count = (send_offset < count) ? count - send_offset : 0;
        if (recv_offset + recv_count > count)
            recv_count = (recv_offset < count) ? count - recv_offset : 0;

        /* Create tensor wrappers for send/recv */
        send_shape[0] = (int)send_count;
        recv_shape[0] = (int)recv_count;

        Tensor send_tensor = {0};
        send_tensor.data  = data + send_offset;
        send_tensor.numel = send_count;
        send_tensor.ndim  = 1;
        send_tensor.shape = send_shape;
        send_tensor.dtype = DTYPE_FLOAT32;

        Tensor recv_tensor = {0};
        recv_tensor.data  = recv_buf;
        recv_tensor.numel = recv_count;
        recv_tensor.ndim  = 1;
        recv_tensor.shape = recv_shape;
        recv_tensor.dtype = DTYPE_FLOAT32;

        /* Send to right neighbor, recv from left neighbor */
        int ret = ops->send(&send_tensor, right, step, ctx);
        if (ret != 0) { free(recv_buf); return -1; }

        ret = ops->recv(&recv_tensor, left, step, ctx);
        if (ret != 0) { free(recv_buf); return -1; }

        /* Reduce received data into local chunk */
        if (recv_count > 0) {
            apply_reduce_op(data + recv_offset, recv_buf, recv_count, op);
        }
    }

    /* Phase 2: All-gather ring */
    for (int step = 0; step < world_size - 1; step++) {
        int send_chunk = (rank - step + 1 + world_size) % world_size;
        int recv_chunk = (rank - step + world_size) % world_size;

        size_t send_offset = (size_t)send_chunk * chunk_size;
        size_t recv_offset = (size_t)recv_chunk * chunk_size;
        size_t send_count  = chunk_size;
        size_t recv_count  = chunk_size;

        if (send_offset + send_count > count)
            send_count = (send_offset < count) ? count - send_offset : 0;
        if (recv_offset + recv_count > count)
            recv_count = (recv_offset < count) ? count - recv_offset : 0;

        send_shape[0] = (int)send_count;
        recv_shape[0] = (int)recv_count;

        Tensor send_tensor = {0};
        send_tensor.data  = data + send_offset;
        send_tensor.numel = send_count;
        send_tensor.ndim  = 1;
        send_tensor.shape = send_shape;
        send_tensor.dtype = DTYPE_FLOAT32;

        Tensor recv_tensor = {0};
        recv_tensor.data  = data + recv_offset;
        recv_tensor.numel = recv_count;
        recv_tensor.ndim  = 1;
        recv_tensor.shape = recv_shape;
        recv_tensor.dtype = DTYPE_FLOAT32;

        int ret = ops->send(&send_tensor, right, world_size + step, ctx);
        if (ret != 0) { free(recv_buf); return -1; }

        ret = ops->recv(&recv_tensor, left, world_size + step, ctx);
        if (ret != 0) { free(recv_buf); return -1; }
    }

    /* Apply averaging if requested */
    if (op == DIST_REDUCE_AVG) {
        float scale = 1.0f / (float)world_size;
        for (size_t i = 0; i < count; i++)
            data[i] *= scale;
    }

    free(recv_buf);
    return 0;
}
