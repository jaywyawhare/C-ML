#include "distributed/data_parallel.h"
#include "distributed/distributed.h"
#include "core/logging.h"
#include "ops/ir/execution.h"
#include "tensor/realize.h"
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

#define DEFAULT_BUCKET_SIZE (25 * 1024 * 1024) /* 25MB in bytes */

/** Default DDP settings: 25MB gradient buckets, buffer broadcast on, no
 * unused-parameter tracking, and copy (not view) gradient semantics. */
DDPConfig cml_ddp_default_config(void) {
    DDPConfig config = {.bucket_size_bytes       = DEFAULT_BUCKET_SIZE,
                        .broadcast_buffers       = true,
                        .find_unused_parameters  = false,
                        .gradient_as_bucket_view = 0};
    return config;
}

/** Wrap @p module for data-parallel training: collect its parameters, broadcast
 * them from rank 0 so every rank starts identical, and lay them out into flat
 * gradient buckets for later all-reduce. Requires cml_dist_init first; returns
 * NULL on error. */
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
    if (ddp->config.gradient_as_bucket_view)
        ddp->grad_is_view = cml_calloc(ddp->num_params, sizeof(bool));

    if (!ddp->buckets || !ddp->bucket_sizes || !ddp->param_to_bucket ||
        (ddp->config.gradient_as_bucket_view && !ddp->grad_is_view)) {
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

/** Run the wrapped module's forward pass, first re-broadcasting buffers from
 * rank 0 when broadcast_buffers is set so all ranks evaluate with equal state. */
Tensor* cml_ddp_forward(CMLDataParallel* ddp, Tensor* input) {
    if (!ddp || !ddp->module || !input)
        return NULL;

    if (ddp->config.broadcast_buffers && ddp->initialized)
        ddp_broadcast_buffers(ddp);

    return module_forward(ddp->module, input);
}

/** Return this rank's slice of @p full_batch along dim 0 as a freshly
 * materialized, caller-owned tensor. Passes the batch through unchanged at
 * world_size <= 1; returns NULL if this rank gets no rows. */
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

/* Repoint `g` at `slot`, which is owned by a gradient bucket, carrying its
 * current values over. Returns false when the tensor cannot be aliased -- device
 * memory, a non-float32 dtype, or a size that doesn't match the slot -- and the
 * caller then falls back to copying.
 *
 * tensor_realize() is the first step because it both materialises a lazy
 * gradient and detaches it from the IR graph: a gradient still wired to a graph
 * node could be re-executed into a fresh buffer later, silently dropping the
 * alias. After the swap the tensor no longer owns its data, so tensor_free()
 * leaves the bucket's memory alone (see the owns_data branch in tensor_free). */
static bool ddp_alias_grad_to_slot(Tensor* g, float* slot, size_t numel) {
    if (!g || !slot || g->numel != numel || numel == 0)
        return false;
    if (g->dtype != DTYPE_FLOAT32)
        return false;
    if (g->buffer_handle)
        return false; /* backend-owned device buffer */
    if (g->device != DEVICE_CPU && g->device != DEVICE_AUTO)
        return false;

    if (tensor_realize(g) != 0 || !g->data)
        return false;
    if ((float*)g->data == slot)
        return true; /* already aliased from a previous step */

    memcpy(slot, g->data, numel * sizeof(float));

    if (g->owns_data) {
        if (g->storage) {
            tensor_storage_release(g); /* shared block: drop our hold */
        } else if (g->from_buffer_cache) {
            cml_buffer_cache_free(g->data, numel * sizeof(float));
        } else {
            cml_free(g->data);
        }
    } else if (g->storage) {
        tensor_storage_release(g);
    }

    g->data              = slot;
    g->owns_data         = false; /* the bucket owns this memory now */
    g->from_buffer_cache = false;
    g->storage           = NULL;
    g->storage_offset    = 0;
    g->is_executed       = true;
    return true;
}

/** True if @p p points inside any of the DDP gradient buckets; used to guard
 * against un-aliasing a gradient that was swapped out for a borrowed view. */
static bool ddp_ptr_in_buckets(const CMLDataParallel* ddp, const void* p) {
    if (!ddp->buckets || !ddp->bucket_sizes)
        return false;
    for (int b = 0; b < ddp->num_buckets; b++) {
        if (!ddp->buckets[b])
            continue;
        const float* lo = ddp->buckets[b];
        const float* hi = lo + ddp->bucket_sizes[b];
        if ((const float*)p >= lo && (const float*)p < hi)
            return true;
    }
    return false;
}

/* Give each aliased gradient its own allocation again, copying the current
 * values out of the bucket. Called before the buckets are released so the
 * module's gradients stay readable after the DDP wrapper is gone. */
static void ddp_unbind_views(CMLDataParallel* ddp) {
    if (!ddp->grad_is_view || !ddp->all_params)
        return;

    for (int i = 0; i < ddp->num_params; i++) {
        if (!ddp->grad_is_view[i])
            continue;
        ddp->grad_is_view[i] = false;

        Parameter* p = ddp->all_params[i];
        Tensor* g    = (p && p->tensor) ? p->tensor->grad : NULL;
        if (!g || !g->data || g->owns_data)
            continue;
        /* Only un-alias a gradient that really points into our buckets. A
         * gradient replaced since the last sync can be a borrowed view of
         * something else entirely, and handing it owns_data would hand it a
         * double free. */
        if (!ddp_ptr_in_buckets(ddp, g->data))
            continue;

        size_t nbytes = g->numel * sizeof(float);
        void* owned   = cml_malloc(nbytes);
        if (!owned) {
            /* Better a gradient-less parameter than one pointing into freed
             * bucket memory. */
            LOG_ERROR("DDP: could not un-alias gradient for param %d", i);
            g->data        = NULL;
            g->is_executed = false;
            continue;
        }
        memcpy(owned, g->data, nbytes);
        g->data      = owned;
        g->owns_data = true;
    }
}

/* Copy bucket `b`'s gradients between the parameter tensors and the flat bucket
 * buffer: `pack` gathers into the bucket, otherwise it scatters back. Gradients
 * may still be lazy (graph autodiff), so each is materialised before its data
 * pointer is touched -- otherwise the sync silently skips it.
 *
 * Under gradient_as_bucket_view the pack pass instead aliases each gradient onto
 * its slot, so both passes become no-ops for every gradient that could be
 * aliased. */
static size_t ddp_bucket_copy(CMLDataParallel* ddp, int b, bool pack) {
    bool view = ddp->config.gradient_as_bucket_view != 0 && ddp->grad_is_view != NULL;
    /* find_unused_parameters keeps a zero-filled slot for a gradient-less param
     * so every rank's bucket layout matches; otherwise the all-reduce would sum
     * mismatched elements. Aliasing needs that same fixed layout: an offset that
     * shifts between steps would leave gradients pointing at the wrong slot. */
    bool reserve  = ddp->config.find_unused_parameters || view;
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
            if (view)
                ddp->grad_is_view[i] = false;
            continue;
        }

        if (ddp->buckets[b]) {
            float* slot = ddp->buckets[b] + offset;
            if (view && pack)
                ddp->grad_is_view[i] = ddp_alias_grad_to_slot(p->tensor->grad, slot, numel);

            /* An aliased gradient already *is* the slot; only the copy-fallback
             * params still need moving in either direction. */
            if (!view || !ddp->grad_is_view[i]) {
                float* grad_data = (float*)p->tensor->grad->data;
                if (pack)
                    memcpy(slot, grad_data, numel * sizeof(float));
                else
                    memcpy(grad_data, slot, numel * sizeof(float));
            }
        }
        offset += numel;
    }
    return offset;
}

/** All-reduce-average the module's gradients across the process group: pack each
 * bucket, reduce it, and scatter the averaged values back. A no-op at
 * world_size 1 unless bucket-view aliasing still needs binding. Returns 0 on
 * success, -1 if DDP is uninitialized. */
int cml_ddp_sync_gradients(CMLDataParallel* ddp) {
    if (!ddp || !ddp->initialized) {
        LOG_ERROR("DDP not initialized");
        return -1;
    }

    int world_size = ddp->group->world_size;
    bool view      = ddp->config.gradient_as_bucket_view != 0 && ddp->grad_is_view != NULL;

    /* There is nothing to reduce against a single rank, but bucket-view aliasing
     * is a storage layout rather than a collective: binding it here keeps
     * single- and multi-process runs on one code path. */
    if (world_size <= 1 && !view)
        return 0;

    LOG_DEBUG("DDP: syncing gradients across %d processes", world_size);

    /* Process each bucket */
    for (int b = 0; b < ddp->num_buckets; b++) {
        if (ddp->bucket_sizes[b] == 0)
            continue;

        /* Gathers into the bucket, or (bucket-view) aliases the gradients onto
         * it so this pass copies nothing. */
        size_t offset = ddp_bucket_copy(ddp, b, true);

        /* All-reduce the bucket */
        if (world_size > 1 && ddp->buckets[b] && offset > 0) {
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

        /* Scatter the reduced values back. Aliased gradients already read
         * straight out of the bucket, and at world_size 1 nothing changed. */
        if (world_size > 1)
            ddp_bucket_copy(ddp, b, false);
    }

    LOG_DEBUG("DDP: gradient sync complete");
    return 0;
}

/** Free the DDP wrapper. Un-aliases any bucket-view gradients first so the
 * module's parameters keep valid gradient storage after the buckets are gone;
 * does not free the wrapped module. */
void cml_ddp_free(CMLDataParallel* ddp) {
    if (!ddp)
        return;

    /* Aliased gradients point into the buckets about to be freed; give them
     * their own storage back first. */
    ddp_unbind_views(ddp);
    cml_free(ddp->grad_is_view);

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
