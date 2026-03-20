#include "tensor/shard.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

static CMLShardedTensor* sharded_tensor_alloc(int num_shards, int ndim, int* shape,
                                               DeviceType* devices, int axis)
{
    CMLShardedTensor* st = (CMLShardedTensor*)calloc(1, sizeof(CMLShardedTensor));
    if (!st) return NULL;

    st->num_shards = num_shards;
    st->axis = axis;
    st->ndim = ndim;

    st->shape = (int*)malloc((size_t)ndim * sizeof(int));
    if (!st->shape) { free(st); return NULL; }
    memcpy(st->shape, shape, (size_t)ndim * sizeof(int));

    st->devices = (DeviceType*)malloc((size_t)num_shards * sizeof(DeviceType));
    if (!st->devices) { free(st->shape); free(st); return NULL; }
    memcpy(st->devices, devices, (size_t)num_shards * sizeof(DeviceType));

    st->shards = (Tensor**)calloc((size_t)num_shards, sizeof(Tensor*));
    if (!st->shards) { free(st->devices); free(st->shape); free(st); return NULL; }

    return st;
}

CMLShardedTensor* tensor_shard(Tensor* t, DeviceType* devices, int num_devices, int axis)
{
    if (!t || !devices || num_devices <= 0) {
        LOG_ERROR("tensor_shard: invalid arguments");
        return NULL;
    }

    int norm_axis = axis < 0 ? t->ndim + axis : axis;
    if (norm_axis < 0 || norm_axis >= t->ndim) {
        LOG_ERROR("tensor_shard: axis %d out of range for ndim %d", axis, t->ndim);
        return NULL;
    }

    int dim_size = t->shape[norm_axis];
    int* sizes = (int*)malloc((size_t)num_devices * sizeof(int));
    if (!sizes) return NULL;

    int base = dim_size / num_devices;
    int remainder = dim_size % num_devices;
    for (int i = 0; i < num_devices; i++) {
        sizes[i] = base + (i < remainder ? 1 : 0);
    }

    CMLShardedTensor* st = tensor_shard_with_sizes(t, devices, num_devices, norm_axis, sizes);
    free(sizes);
    return st;
}

CMLShardedTensor* tensor_shard_with_sizes(Tensor* t, DeviceType* devices, int num_devices,
                                           int axis, int* sizes)
{
    if (!t || !devices || !sizes || num_devices <= 0) {
        LOG_ERROR("tensor_shard_with_sizes: invalid arguments");
        return NULL;
    }

    int norm_axis = axis < 0 ? t->ndim + axis : axis;
    if (norm_axis < 0 || norm_axis >= t->ndim) {
        LOG_ERROR("tensor_shard_with_sizes: axis %d out of range", axis);
        return NULL;
    }

    int total = 0;
    for (int i = 0; i < num_devices; i++) total += sizes[i];
    if (total != t->shape[norm_axis]) {
        LOG_ERROR("tensor_shard_with_sizes: sizes sum %d != dim size %d", total, t->shape[norm_axis]);
        return NULL;
    }

    tensor_ensure_executed(t);
    const float* src = (const float*)tensor_data_ptr(t);
    if (!src) {
        LOG_ERROR("tensor_shard_with_sizes: failed to get tensor data");
        return NULL;
    }

    CMLShardedTensor* st = sharded_tensor_alloc(num_devices, t->ndim, t->shape, devices, norm_axis);
    if (!st) return NULL;

    size_t outer = 1;
    for (int i = 0; i < norm_axis; i++) outer *= (size_t)t->shape[i];

    size_t inner = 1;
    for (int i = norm_axis + 1; i < t->ndim; i++) inner *= (size_t)t->shape[i];

    int full_axis_size = t->shape[norm_axis];
    int offset = 0;

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    for (int s = 0; s < num_devices; s++) {
        int shard_axis = sizes[s];
        size_t shard_elems = outer * (size_t)shard_axis * inner;

        float* shard_data = (float*)malloc(shard_elems * sizeof(float));
        if (!shard_data) {
            LOG_ERROR("tensor_shard_with_sizes: allocation failed for shard %d", s);
            sharded_tensor_free(st);
            return NULL;
        }

        for (size_t o = 0; o < outer; o++) {
            const float* row_src = src + o * (size_t)full_axis_size * inner + (size_t)offset * inner;
            float* row_dst = shard_data + o * (size_t)shard_axis * inner;
            memcpy(row_dst, row_src, (size_t)shard_axis * inner * sizeof(float));
        }

        int* shard_shape = (int*)malloc((size_t)t->ndim * sizeof(int));
        if (!shard_shape) {
            free(shard_data);
            sharded_tensor_free(st);
            return NULL;
        }
        memcpy(shard_shape, t->shape, (size_t)t->ndim * sizeof(int));
        shard_shape[norm_axis] = shard_axis;

        cfg.device = devices[s];
        st->shards[s] = tensor_from_data(shard_data, shard_shape, t->ndim, &cfg);
        free(shard_data);
        free(shard_shape);

        if (!st->shards[s]) {
            LOG_ERROR("tensor_shard_with_sizes: failed to create shard %d", s);
            sharded_tensor_free(st);
            return NULL;
        }

        offset += shard_axis;
    }

    return st;
}

Tensor* tensor_unshard(CMLShardedTensor* st)
{
    if (!st || !st->shards || st->num_shards <= 0) {
        LOG_ERROR("tensor_unshard: invalid arguments");
        return NULL;
    }

    int axis = st->axis;
    int ndim = st->ndim;

    size_t outer = 1;
    for (int i = 0; i < axis; i++) outer *= (size_t)st->shape[i];

    size_t inner = 1;
    for (int i = axis + 1; i < ndim; i++) inner *= (size_t)st->shape[i];

    size_t total_elems = outer * (size_t)st->shape[axis] * inner;
    float* out_data = (float*)malloc(total_elems * sizeof(float));
    if (!out_data) {
        LOG_ERROR("tensor_unshard: allocation failed");
        return NULL;
    }

    int offset = 0;
    int full_axis_size = st->shape[axis];

    for (int s = 0; s < st->num_shards; s++) {
        Tensor* shard = st->shards[s];
        tensor_ensure_executed(shard);
        const float* sdata = (const float*)tensor_data_ptr(shard);
        if (!sdata) {
            LOG_ERROR("tensor_unshard: failed to get data for shard %d", s);
            free(out_data);
            return NULL;
        }

        int shard_axis = shard->shape[axis];
        for (size_t o = 0; o < outer; o++) {
            float* row_dst = out_data + o * (size_t)full_axis_size * inner + (size_t)offset * inner;
            const float* row_src = sdata + o * (size_t)shard_axis * inner;
            memcpy(row_dst, row_src, (size_t)shard_axis * inner * sizeof(float));
        }

        offset += shard_axis;
    }

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* result = tensor_from_data(out_data, st->shape, ndim, &cfg);
    free(out_data);
    return result;
}

void sharded_tensor_free(CMLShardedTensor* st)
{
    if (!st) return;
    if (st->shards) {
        for (int i = 0; i < st->num_shards; i++) {
            if (st->shards[i]) tensor_free(st->shards[i]);
        }
        free(st->shards);
    }
    free(st->shape);
    free(st->devices);
    free(st);
}

CMLShardedTensor* sharded_add(CMLShardedTensor* a, CMLShardedTensor* b)
{
    if (!a || !b) {
        LOG_ERROR("sharded_add: NULL argument");
        return NULL;
    }
    if (a->num_shards != b->num_shards) {
        LOG_ERROR("sharded_add: shard count mismatch (%d vs %d)", a->num_shards, b->num_shards);
        return NULL;
    }
    if (a->axis != b->axis) {
        LOG_ERROR("sharded_add: axis mismatch (%d vs %d)", a->axis, b->axis);
        return NULL;
    }

    CMLShardedTensor* result = sharded_tensor_alloc(a->num_shards, a->ndim, a->shape,
                                                     a->devices, a->axis);
    if (!result) return NULL;

    for (int s = 0; s < a->num_shards; s++) {
        Tensor* sa = a->shards[s];
        Tensor* sb = b->shards[s];
        tensor_ensure_executed(sa);
        tensor_ensure_executed(sb);

        const float* da = (const float*)tensor_data_ptr(sa);
        const float* db = (const float*)tensor_data_ptr(sb);
        if (!da || !db) {
            LOG_ERROR("sharded_add: failed to get data for shard %d", s);
            sharded_tensor_free(result);
            return NULL;
        }

        size_t n = sa->numel;
        float* out = (float*)malloc(n * sizeof(float));
        if (!out) {
            sharded_tensor_free(result);
            return NULL;
        }

        for (size_t i = 0; i < n; i++) out[i] = da[i] + db[i];

        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = a->devices[s],
                            .has_dtype = true, .has_device = true};
        result->shards[s] = tensor_from_data(out, sa->shape, sa->ndim, &cfg);
        free(out);

        if (!result->shards[s]) {
            sharded_tensor_free(result);
            return NULL;
        }
    }

    return result;
}

CMLShardedTensor* sharded_allreduce_sum(CMLShardedTensor* st)
{
    if (!st || st->num_shards <= 0) {
        LOG_ERROR("sharded_allreduce_sum: invalid arguments");
        return NULL;
    }

    Tensor* first = st->shards[0];
    tensor_ensure_executed(first);
    size_t numel = first->numel;

    float* sum_data = (float*)calloc(numel, sizeof(float));
    if (!sum_data) return NULL;

    for (int s = 0; s < st->num_shards; s++) {
        Tensor* shard = st->shards[s];
        tensor_ensure_executed(shard);
        const float* sd = (const float*)tensor_data_ptr(shard);
        if (!sd) {
            free(sum_data);
            return NULL;
        }
        for (size_t i = 0; i < numel; i++) sum_data[i] += sd[i];
    }

    CMLShardedTensor* result = sharded_tensor_alloc(st->num_shards, first->ndim, first->shape,
                                                     st->devices, st->axis);
    if (!result) { free(sum_data); return NULL; }

    for (int s = 0; s < st->num_shards; s++) {
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = st->devices[s],
                            .has_dtype = true, .has_device = true};
        result->shards[s] = tensor_from_data(sum_data, first->shape, first->ndim, &cfg);
        if (!result->shards[s]) {
            free(sum_data);
            sharded_tensor_free(result);
            return NULL;
        }
    }

    free(sum_data);
    return result;
}

static void matmul_2d(const float* a, int M, int K, const float* b, int N, float* out)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}

CMLShardedTensor* sharded_matmul(CMLShardedTensor* a, CMLShardedTensor* b)
{
    if (!a || !b) {
        LOG_ERROR("sharded_matmul: NULL argument");
        return NULL;
    }
    if (a->ndim != 2 || b->ndim != 2) {
        LOG_ERROR("sharded_matmul: only 2-D tensors supported");
        return NULL;
    }
    if (a->num_shards != b->num_shards) {
        LOG_ERROR("sharded_matmul: shard count mismatch");
        return NULL;
    }

    int num_shards = a->num_shards;
    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    if (a->axis == 1 && b->axis == 0) {
        /* A sharded on columns, B sharded on rows: partial products need allreduce */
        int out_shape[2] = {M, N};

        CMLShardedTensor* partial = sharded_tensor_alloc(num_shards, 2, out_shape,
                                                          a->devices, 0);
        if (!partial) return NULL;

        for (int s = 0; s < num_shards; s++) {
            Tensor* sa = a->shards[s];
            Tensor* sb = b->shards[s];
            tensor_ensure_executed(sa);
            tensor_ensure_executed(sb);

            const float* da = (const float*)tensor_data_ptr(sa);
            const float* db = (const float*)tensor_data_ptr(sb);
            if (!da || !db) {
                sharded_tensor_free(partial);
                return NULL;
            }

            int local_K = sa->shape[1];
            float* out = (float*)malloc((size_t)M * N * sizeof(float));
            if (!out) { sharded_tensor_free(partial); return NULL; }

            matmul_2d(da, M, local_K, db, N, out);

            TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = a->devices[s],
                                .has_dtype = true, .has_device = true};
            partial->shards[s] = tensor_from_data(out, out_shape, 2, &cfg);
            free(out);

            if (!partial->shards[s]) {
                sharded_tensor_free(partial);
                return NULL;
            }
        }

        CMLShardedTensor* result = sharded_allreduce_sum(partial);
        sharded_tensor_free(partial);
        return result;

    } else if (a->axis == 0) {
        /* A sharded on rows: independent matmuls, result sharded on rows */
        int out_shape[2] = {M, N};
        CMLShardedTensor* result = sharded_tensor_alloc(num_shards, 2, out_shape,
                                                         a->devices, 0);
        if (!result) return NULL;

        for (int s = 0; s < num_shards; s++) {
            Tensor* sa = a->shards[s];
            Tensor* sb = b->shards[s];
            tensor_ensure_executed(sa);
            tensor_ensure_executed(sb);

            const float* da = (const float*)tensor_data_ptr(sa);
            const float* db = (const float*)tensor_data_ptr(sb);
            if (!da || !db) {
                sharded_tensor_free(result);
                return NULL;
            }

            int local_M = sa->shape[0];
            int shard_out_shape[2] = {local_M, N};
            float* out = (float*)malloc((size_t)local_M * N * sizeof(float));
            if (!out) { sharded_tensor_free(result); return NULL; }

            matmul_2d(da, local_M, K, db, N, out);

            TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = a->devices[s],
                                .has_dtype = true, .has_device = true};
            result->shards[s] = tensor_from_data(out, shard_out_shape, 2, &cfg);
            free(out);

            if (!result->shards[s]) {
                sharded_tensor_free(result);
                return NULL;
            }
        }

        return result;

    } else {
        LOG_ERROR("sharded_matmul: unsupported sharding config (a.axis=%d, b.axis=%d)",
                  a->axis, b->axis);
        return NULL;
    }
}

CMLShardedTensor* tensor_replicate(Tensor* t, DeviceType* devices, int num_devices)
{
    if (!t || !devices || num_devices <= 0) {
        LOG_ERROR("tensor_replicate: invalid arguments");
        return NULL;
    }

    tensor_ensure_executed(t);
    const float* src = (const float*)tensor_data_ptr(t);
    if (!src) {
        LOG_ERROR("tensor_replicate: failed to get tensor data");
        return NULL;
    }

    CMLShardedTensor* st = sharded_tensor_alloc(num_devices, t->ndim, t->shape, devices, 0);
    if (!st) return NULL;

    for (int i = 0; i < num_devices; i++) {
        TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = devices[i],
                            .has_dtype = true, .has_device = true};
        st->shards[i] = tensor_from_data(src, t->shape, t->ndim, &cfg);
        if (!st->shards[i]) {
            sharded_tensor_free(st);
            return NULL;
        }
    }

    return st;
}
