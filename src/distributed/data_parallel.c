#include "distributed/data_parallel.h"
#include "distributed/distributed.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

#define DEFAULT_BUCKET_SIZE (25 * 1024 * 1024) /* 25MB in bytes */

DDPConfig cml_ddp_default_config(void) {
    DDPConfig config = {.bucket_size_bytes       = DEFAULT_BUCKET_SIZE,
                        .broadcast_buffers       = true,
                        .find_unused_parameters  = false,
                        .gradient_as_bucket_view = 0};
    return config;
}

CMLDataParallel* cml_ddp_create(Module* module, const DDPConfig* config) {
    if (!module) {
        LOG_ERROR("NULL module for DDP");
        return NULL;
    }

    if (!cml_dist_is_initialized()) {
        LOG_ERROR("Distributed not initialized - call cml_dist_init first");
        return NULL;
    }

    CMLDataParallel* ddp = cml_calloc(1, sizeof(CMLDataParallel));
    if (!ddp)
        return NULL;

    ddp->module = module;
    ddp->group  = cml_dist_get_default_group();
    ddp->config = config ? *config : cml_ddp_default_config();

    /* Collect all parameters */
    int result = module_collect_parameters(module, &ddp->all_params, &ddp->num_params, true);
    if (result != 0 || ddp->num_params == 0) {
        LOG_WARNING("DDP: no parameters found in module");
        cml_free(ddp);
        return NULL;
    }

    /* Broadcast parameters from rank 0 */
    LOG_INFO("DDP: broadcasting %d parameters from rank 0", ddp->num_params);
    for (int i = 0; i < ddp->num_params; i++) {
        if (ddp->all_params[i] && ddp->all_params[i]->tensor) {
            cml_dist_broadcast(ddp->all_params[i]->tensor, 0);
        }
    }

    /* Setup gradient buckets */
    size_t bucket_size_floats = ddp->config.bucket_size_bytes / sizeof(float);
    size_t total_params_size  = 0;

    for (int i = 0; i < ddp->num_params; i++) {
        if (ddp->all_params[i] && ddp->all_params[i]->tensor)
            total_params_size += ddp->all_params[i]->tensor->numel;
    }

    ddp->num_buckets = (int)((total_params_size + bucket_size_floats - 1) / bucket_size_floats);
    if (ddp->num_buckets < 1)
        ddp->num_buckets = 1;

    ddp->buckets         = cml_calloc(ddp->num_buckets, sizeof(float*));
    ddp->bucket_sizes    = cml_calloc(ddp->num_buckets, sizeof(size_t));
    ddp->param_to_bucket = cml_calloc(ddp->num_params, sizeof(int));

    if (!ddp->buckets || !ddp->bucket_sizes || !ddp->param_to_bucket) {
        cml_ddp_free(ddp);
        return NULL;
    }

    /* Assign parameters to buckets */
    size_t current_size = 0;
    int current_bucket  = 0;

    for (int i = 0; i < ddp->num_params; i++) {
        ddp->param_to_bucket[i] = current_bucket;

        if (ddp->all_params[i] && ddp->all_params[i]->tensor) {
            size_t param_size = ddp->all_params[i]->tensor->numel;
            ddp->bucket_sizes[current_bucket] += param_size;
            current_size += param_size;

            if (current_size >= bucket_size_floats && current_bucket < ddp->num_buckets - 1) {
                current_bucket++;
                current_size = 0;
            }
        }
    }

    /* Allocate bucket buffers */
    for (int b = 0; b < ddp->num_buckets; b++) {
        if (ddp->bucket_sizes[b] > 0) {
            ddp->buckets[b] = cml_calloc(ddp->bucket_sizes[b], sizeof(float));
            if (!ddp->buckets[b]) {
                LOG_ERROR("DDP: failed to allocate bucket %d", b);
            }
        }
    }

    ddp->initialized = true;

    LOG_INFO("DDP initialized: %d params, %d buckets, world_size=%d", ddp->num_params,
             ddp->num_buckets, ddp->group->world_size);

    return ddp;
}

/* Broadcast every registered non-trainable buffer (e.g. BatchNorm running
 * stats) from rank 0 so all ranks evaluate with identical state. Walks the
 * Module->next chain; containers flatten children via ->next. */
static void ddp_broadcast_buffers(CMLDataParallel* ddp) {
    int world_size = ddp->group ? ddp->group->world_size : 1;
    if (world_size <= 1)
        return;

    int count = 0;
    for (Module* m = ddp->module; m; m = m->next) {
        for (int i = 0; i < m->num_buffers; i++) {
            if (m->buffers[i])
                cml_dist_broadcast(m->buffers[i], 0);
            count++;
        }
    }
    if (count > 0)
        LOG_DEBUG("DDP: broadcast %d buffers from rank 0", count);
}

Tensor* cml_ddp_forward(CMLDataParallel* ddp, Tensor* input) {
    if (!ddp || !ddp->module || !input)
        return NULL;

    if (ddp->config.broadcast_buffers && ddp->initialized)
        ddp_broadcast_buffers(ddp);

    return module_forward(ddp->module, input);
}

Tensor* cml_ddp_shard_input(CMLDataParallel* ddp, Tensor* full_batch) {
    if (!ddp || !full_batch || full_batch->ndim < 1)
        return full_batch;

    int ws   = ddp->group ? ddp->group->world_size : 1;
    int rank = ddp->group ? ddp->group->rank : 0;
    if (ws <= 1)
        return full_batch; /* nothing to shard */

    /* Split the batch (dim 0) across ranks; the first `rem` ranks take one extra
     * row so all rows are covered when B isn't divisible by world_size. Returns
     * a fresh materialized tensor holding just this rank's rows — the caller owns
     * it and should free it. Without this every rank trained on the full batch. */
    int B     = full_batch->shape[0];
    int base  = B / ws;
    int rem   = B % ws;
    int start = rank * base + (rank < rem ? rank : rem);
    int count = base + (rank < rem ? 1 : 0);
    if (count <= 0) {
        LOG_WARNING("DDP: rank %d has no rows for batch size %d / world_size %d", rank, B, ws);
        return NULL;
    }

    tensor_ensure_executed(full_batch);
    const float* src = (const float*)tensor_data_ptr(full_batch);
    if (!src)
        return NULL;

    size_t row = 1;
    for (int d = 1; d < full_batch->ndim; d++)
        row *= (size_t)full_batch->shape[d];

    float* dst = (float*)cml_malloc((size_t)count * row * sizeof(float));
    int* shape = (int*)cml_malloc((size_t)full_batch->ndim * sizeof(int));
    if (!dst || !shape) {
        cml_free(dst);
        cml_free(shape);
        return NULL;
    }

    memcpy(dst, src + (size_t)start * row, (size_t)count * row * sizeof(float));
    shape[0] = count;
    for (int d = 1; d < full_batch->ndim; d++)
        shape[d] = full_batch->shape[d];

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* shard = tensor_from_data(dst, shape, full_batch->ndim, &cfg);
    cml_free(dst);
    cml_free(shape);
    return shard;
}

/* Copy bucket `b`'s gradients between the parameter tensors and the flat bucket
 * buffer: `pack` gathers into the bucket, otherwise it scatters back. Gradients
 * may still be lazy (graph autodiff), so each is materialised before its data
 * pointer is touched -- otherwise the sync silently skips it. */
static size_t ddp_bucket_copy(CMLDataParallel* ddp, int b, bool pack) {
    /* find_unused_parameters keeps a zero-filled slot for a gradient-less param
     * so every rank's bucket layout matches; otherwise the all-reduce would sum
     * mismatched elements. */
    bool reserve  = ddp->config.find_unused_parameters;
    size_t offset = 0;
    for (int i = 0; i < ddp->num_params; i++) {
        if (ddp->param_to_bucket[i] != b)
            continue;

        Parameter* p = ddp->all_params[i];
        size_t numel = (p && p->tensor) ? p->tensor->numel : 0;

        bool has_grad = p && p->tensor && p->tensor->grad;
        if (has_grad) {
            tensor_ensure_executed(p->tensor->grad);
            has_grad = (p->tensor->grad->data != NULL);
        }

        if (!has_grad) {
            if (reserve && numel > 0 && ddp->buckets[b]) {
                if (pack)
                    memset(ddp->buckets[b] + offset, 0, numel * sizeof(float));
                offset += numel;
            }
            continue;
        }

        float* grad_data = (float*)p->tensor->grad->data;
        if (ddp->buckets[b]) {
            if (pack)
                memcpy(ddp->buckets[b] + offset, grad_data, numel * sizeof(float));
            else
                memcpy(grad_data, ddp->buckets[b] + offset, numel * sizeof(float));
        }
        offset += numel;
    }
    return offset;
}

int cml_ddp_sync_gradients(CMLDataParallel* ddp) {
    if (!ddp || !ddp->initialized) {
        LOG_ERROR("DDP not initialized");
        return -1;
    }

    int world_size = ddp->group->world_size;
    if (world_size <= 1) {
        /* No need to sync in single-process mode */
        return 0;
    }

    LOG_DEBUG("DDP: syncing gradients across %d processes", world_size);

    /* Process each bucket */
    for (int b = 0; b < ddp->num_buckets; b++) {
        if (ddp->bucket_sizes[b] == 0)
            continue;

        size_t offset = ddp_bucket_copy(ddp, b, true);

        /* All-reduce the bucket */
        if (ddp->buckets[b] && offset > 0) {
            /* Create a temporary tensor for the bucket */
            int shape[1]         = {(int)offset};
            Tensor bucket_tensor = {.data      = ddp->buckets[b],
                                    .shape     = shape,
                                    .ndim      = 1,
                                    .numel     = offset,
                                    .dtype     = DTYPE_FLOAT32,
                                    .device    = DEVICE_CPU,
                                    .owns_data = false};

            cml_dist_allreduce(&bucket_tensor, DIST_REDUCE_SUM);

            /* Average by world_size */
            float scale = 1.0f / (float)world_size;
            for (size_t j = 0; j < offset; j++)
                ddp->buckets[b][j] *= scale;
        }

        ddp_bucket_copy(ddp, b, false);
    }

    LOG_DEBUG("DDP: gradient sync complete");
    return 0;
}

void cml_ddp_free(CMLDataParallel* ddp) {
    if (!ddp)
        return;

    if (ddp->buckets) {
        for (int b = 0; b < ddp->num_buckets; b++)
            cml_free(ddp->buckets[b]);
        cml_free(ddp->buckets);
    }

    cml_free(ddp->bucket_sizes);
    cml_free(ddp->param_to_bucket);
    cml_free(ddp->all_params);
    cml_free(ddp);
}
