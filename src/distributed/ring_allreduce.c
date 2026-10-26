#include "distributed/ring_allreduce.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "alloc/cml_allocator.h"

typedef struct {
    DistCommOps* ops;
    Tensor* tensor;
    int peer;
    int tag;
    void* ctx;
    int rc;
} RingSendArg;

/** Thread body: send one chunk to the right neighbour, concurrently with the main-thread recv.
 * @param arg (void*)
 * @return (void*)
 */
static void* ring_send_thread(void* arg) {
    RingSendArg* a = (RingSendArg*)arg;
    a->rc          = a->ops->send(a->tensor, a->peer, a->tag, a->ctx);
    return NULL;
}

/**
 * Full-duplex ring step: send to the right neighbour and receive from the left at the same time,
 * so a step costs max(send, recv) instead of send + recv. The two directions use different
 * sockets and disjoint buffers, so overlapping them is safe and deadlock-free (every rank is
 * simultaneously sending and receiving). Falls back to serial send-then-recv if no thread.
 * @param ops (DistCommOps*)
 * @param send_t (Tensor*)
 * @param right (int)
 * @param recv_t (Tensor*)
 * @param left (int)
 * @param tag (int)
 * @param ctx (void*)
 * @return (int)
 */
int cml_ring_sendrecv(DistCommOps* ops, Tensor* send_t, int right, Tensor* recv_t, int left,
                      int tag, void* ctx) {
    RingSendArg sa = {ops, send_t, right, tag, ctx, 0};
    pthread_t th;
    if (pthread_create(&th, NULL, ring_send_thread, &sa) != 0) {
        int r = ops->send(send_t, right, tag, ctx);
        if (r != 0)
            return r;
        return ops->recv(recv_t, left, tag, ctx);
    }
    int rc = ops->recv(recv_t, left, tag, ctx);
    pthread_join(th, NULL);
    return (rc != 0 || sa.rc != 0) ? -1 : 0;
}

/**
 * Fold @p src into @p dst element-wise under @p op (sum/avg accumulate; avg is
 * scaled later once, after the ring completes).
 * @param dst (float*)
 * @param src (const float*)
 * @param n (size_t)
 * @param op (DistReduceOp)
 * @return (void)
 */
static void apply_reduce_op(float* dst, const float* src, size_t n, DistReduceOp op) {
    /* Switch hoisted out of the loop so each branch is a tight, auto-vectorizable kernel. */
    switch (op) {
    case DIST_REDUCE_SUM:
    case DIST_REDUCE_AVG:
        for (size_t i = 0; i < n; i++)
            dst[i] += src[i];
        break;
    case DIST_REDUCE_PRODUCT:
        for (size_t i = 0; i < n; i++)
            dst[i] *= src[i];
        break;
    case DIST_REDUCE_MAX:
        for (size_t i = 0; i < n; i++)
            dst[i] = src[i] > dst[i] ? src[i] : dst[i];
        break;
    case DIST_REDUCE_MIN:
        for (size_t i = 0; i < n; i++)
            dst[i] = src[i] < dst[i] ? src[i] : dst[i];
        break;
    }
}

/**
 * Bandwidth-optimal ring all-reduce of @p data in place: a reduce-scatter phase
 * followed by an all-gather phase, each world_size-1 steps over the @p ops
 * send/recv pair. @p op's averaging, if any, is applied once at the end. Returns
 * 0 on success, -1 on bad args or a transport error.
 * @param data (float*)
 * @param count (size_t)
 * @param world_size (int)
 * @param rank (int)
 * @param op (DistReduceOp)
 * @param ops (DistCommOps*)
 * @param ctx (void*)
 * @return (int)
 */
int cml_ring_allreduce(float* data, size_t count, int world_size, int rank, DistReduceOp op,
                       DistCommOps* ops, void* ctx) {
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
    float* recv_buf   = (float*)cml_malloc(chunk_size * sizeof(float));
    if (!recv_buf)
        return -1;

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
        send_tensor.data   = data + send_offset;
        send_tensor.numel  = send_count;
        send_tensor.ndim   = 1;
        send_tensor.shape  = send_shape;
        send_tensor.dtype  = DTYPE_FLOAT32;

        Tensor recv_tensor = {0};
        recv_tensor.data   = recv_buf;
        recv_tensor.numel  = recv_count;
        recv_tensor.ndim   = 1;
        recv_tensor.shape  = recv_shape;
        recv_tensor.dtype  = DTYPE_FLOAT32;

        /* Send to right neighbor, recv from left neighbor (parity-ordered) */
        int ret = cml_ring_sendrecv(ops, &send_tensor, right, &recv_tensor, left, step, ctx);
        if (ret != 0) {
            cml_free(recv_buf);
            return -1;
        }

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
        send_tensor.data   = data + send_offset;
        send_tensor.numel  = send_count;
        send_tensor.ndim   = 1;
        send_tensor.shape  = send_shape;
        send_tensor.dtype  = DTYPE_FLOAT32;

        Tensor recv_tensor = {0};
        recv_tensor.data   = data + recv_offset;
        recv_tensor.numel  = recv_count;
        recv_tensor.ndim   = 1;
        recv_tensor.shape  = recv_shape;
        recv_tensor.dtype  = DTYPE_FLOAT32;

        int ret =
            cml_ring_sendrecv(ops, &send_tensor, right, &recv_tensor, left, world_size + step, ctx);
        if (ret != 0) {
            cml_free(recv_buf);
            return -1;
        }
    }

    /* Apply averaging if requested */
    if (op == DIST_REDUCE_AVG) {
        float scale = 1.0f / (float)world_size;
        for (size_t i = 0; i < count; i++)
            data[i] *= scale;
    }

    cml_free(recv_buf);
    return 0;
}
