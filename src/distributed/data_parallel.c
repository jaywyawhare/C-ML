/**
 * @file data_parallel.c
 * @brief Distributed Data Parallel (DDP) implementation
 *
 * PyTorch-style DDP: broadcast params from rank 0, bucketed gradient
 * all-reduce (25MB buckets), average by world_size.
 */

#include "distributed/data_parallel.h"
#include "distributed/distributed.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

#define DEFAULT_BUCKET_SIZE (25 * 1024 * 1024) /* 25MB in bytes */

DDPConfig cml_ddp_default_config(void) {
    DDPConfig config = {
        .bucket_size_bytes = DEFAULT_BUCKET_SIZE,
        .broadcast_buffers = true,
        .find_unused_parameters = false,
        .gradient_as_bucket_view = 0
    };
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

    CMLDataParallel* ddp = calloc(1, sizeof(CMLDataParallel));
    if (!ddp)
        return NULL;

    ddp->module = module;
    ddp->group = cml_dist_get_default_group();
    ddp->config = config ? *config : cml_ddp_default_config();

    /* Collect all parameters */
    int result = module_collect_parameters(module, &ddp->all_params,
                                           &ddp->num_params, true);
    if (result != 0 || ddp->num_params == 0) {
        LOG_WARNING("DDP: no parameters found in module");
        free(ddp);
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
    size_t total_params_size = 0;

    for (int i = 0; i < ddp->num_params; i++) {
        if (ddp->all_params[i] && ddp->all_params[i]->tensor)
            total_params_size += ddp->all_params[i]->tensor->numel;
    }

    ddp->num_buckets = (int)((total_params_size + bucket_size_floats - 1) / bucket_size_floats);
    if (ddp->num_buckets < 1)
        ddp->num_buckets = 1;

    ddp->buckets = calloc(ddp->num_buckets, sizeof(float*));
    ddp->bucket_sizes = calloc(ddp->num_buckets, sizeof(size_t));
    ddp->bucket_ready = calloc(ddp->num_buckets, sizeof(bool));
    ddp->param_to_bucket = calloc(ddp->num_params, sizeof(int));

    if (!ddp->buckets || !ddp->bucket_sizes || !ddp->bucket_ready || !ddp->param_to_bucket) {
        cml_ddp_free(ddp);
        return NULL;
    }

    /* Assign parameters to buckets */
    size_t current_size = 0;
    int current_bucket = 0;

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
            ddp->buckets[b] = calloc(ddp->bucket_sizes[b], sizeof(float));
            if (!ddp->buckets[b]) {
                LOG_ERROR("DDP: failed to allocate bucket %d", b);
            }
        }
    }

    ddp->initialized = true;

    LOG_INFO("DDP initialized: %d params, %d buckets, world_size=%d",
             ddp->num_params, ddp->num_buckets, ddp->group->world_size);

    return ddp;
}

Tensor* cml_ddp_forward(CMLDataParallel* ddp, Tensor* input) {
    if (!ddp || !ddp->module || !input)
        return NULL;

    return module_forward(ddp->module, input);
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

        /* Pack gradients into bucket */
        size_t offset = 0;
        for (int i = 0; i < ddp->num_params; i++) {
            if (ddp->param_to_bucket[i] != b)
                continue;

            Parameter* p = ddp->all_params[i];
            if (!p || !p->tensor || !p->tensor->grad || !p->tensor->grad->data)
                continue;

            float* grad_data = (float*)p->tensor->grad->data;
            size_t numel = p->tensor->numel;

            if (ddp->buckets[b]) {
                memcpy(ddp->buckets[b] + offset, grad_data, numel * sizeof(float));
            }
            offset += numel;
        }

        /* All-reduce the bucket */
        if (ddp->buckets[b] && offset > 0) {
            /* Create a temporary tensor for the bucket */
            int shape[1] = {(int)offset};
            TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                                .has_dtype = true, .has_device = true};
            Tensor bucket_tensor = {
                .data = ddp->buckets[b],
                .shape = shape,
                .ndim = 1,
                .numel = offset,
                .dtype = DTYPE_FLOAT32,
                .device = DEVICE_CPU,
                .owns_data = false
            };

            cml_dist_allreduce(&bucket_tensor, DIST_REDUCE_SUM);

            /* Average by world_size */
            float scale = 1.0f / (float)world_size;
            for (size_t j = 0; j < offset; j++)
                ddp->buckets[b][j] *= scale;
        }

        /* Unpack gradients from bucket */
        offset = 0;
        for (int i = 0; i < ddp->num_params; i++) {
            if (ddp->param_to_bucket[i] != b)
                continue;

            Parameter* p = ddp->all_params[i];
            if (!p || !p->tensor || !p->tensor->grad || !p->tensor->grad->data)
                continue;

            float* grad_data = (float*)p->tensor->grad->data;
            size_t numel = p->tensor->numel;

            if (ddp->buckets[b]) {
                memcpy(grad_data, ddp->buckets[b] + offset, numel * sizeof(float));
            }
            offset += numel;
        }
    }

    LOG_DEBUG("DDP: gradient sync complete");
    return 0;
}

void cml_ddp_free(CMLDataParallel* ddp) {
    if (!ddp)
        return;

    if (ddp->buckets) {
        for (int b = 0; b < ddp->num_buckets; b++)
            free(ddp->buckets[b]);
        free(ddp->buckets);
    }

    free(ddp->bucket_sizes);
    free(ddp->bucket_ready);
    free(ddp->param_to_bucket);
    free(ddp->all_params);
    free(ddp);
}
