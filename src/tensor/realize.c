#include "tensor/realize.h"
#include "ops/ir/execution.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

/** True once `t` holds materialized data (non-NULL ->data). */
bool tensor_is_realized(const Tensor* t) { return t != NULL && t->data != NULL; }

/** Execute the IR graph up to `t`, copy any borrowed plan buffer into owned
 *  storage, and detach `t` from the graph so it survives graph teardown (the
 *  linkage is saved for tensor_unrealize). Returns 0 on success, nonzero on error. */
int tensor_realize(Tensor* t) {
    if (!t)
        return -1;

    /* If already detached from the IR graph and has data, nothing to do. */
    if (!t->ir_node && t->data)
        return 0;

    /* Execute the IR graph up to this node (allocates t->data). */
    int ret = tensor_ensure_executed(t);
    if (ret != 0)
        return ret;

    /* If data is borrowed from an execution plan buffer (owns_data==false),
     * copy into a new owned allocation so this tensor survives plan eviction
     * by cml_graph_cache_reset_global() / cml_ir_reset_global_context(). */
    if (t->data && !t->owns_data) {
        size_t nbytes = t->numel * cml_dtype_size(t->dtype);
        void* owned   = cml_malloc(nbytes);
        if (owned) {
            memcpy(owned, t->data, nbytes);
            t->data      = owned;
            t->owns_data = true;
            /* Drop any shared-storage hold (pinned view): it now owns a copy. */
            tensor_storage_release(t);
        }
    }

    /* Detach from the IR graph so this tensor survives cml_ir_free().
     * Save the linkage first so tensor_unrealize() can reconnect for
     * gradient checkpointing / re-materialization. */
    if (t->ir_node) {
        t->saved_ir_node    = t->ir_node;
        t->saved_ir_context = t->ir_context;
        t->ir_node->output  = NULL;
        t->ir_node          = NULL;
    }
    t->ir_context = NULL;

    return 0;
}

/** Execute every unrealized tensor in the array; returns the last error code, or 0. */
int tensor_realize_all(Tensor** tensors, int num_tensors) {
    if (!tensors || num_tensors <= 0)
        return -1;
    int rc = 0;
    for (int i = 0; i < num_tensors; ++i) {
        if (!tensors[i] || tensor_is_realized(tensors[i]))
            continue;
        int r = tensor_ensure_executed(tensors[i]);
        if (r != 0)
            rc = r;
    }
    return rc;
}

/** Free `t`'s materialized data and reconnect it to its saved IR node so it can
 *  be recomputed on demand (gradient checkpointing / re-materialization). */
void tensor_unrealize(Tensor* t) {
    if (!t || !t->data)
        return;
    if (t->owns_data) {
        if (t->storage) {
            /* Shared block: views may still read it after we detach. */
            tensor_storage_release(t);
        } else if (t->from_buffer_cache) {
            cml_buffer_cache_free(t->data, t->numel * cml_dtype_size(t->dtype));
        } else {
            cml_free(t->data);
        }
    }
    t->data        = NULL;
    t->is_executed = false;
    t->owns_data   = false;

    /* Reconnect to the IR graph so this tensor can be re-materialized.
     * Required for gradient checkpointing: free activations during the
     * forward pass, recompute on demand during backward. */
    if (t->saved_ir_node && !t->ir_node) {
        t->saved_ir_node->output      = t;
        t->saved_ir_node->is_executed = false;
        t->ir_node                    = t->saved_ir_node;
        t->ir_context                 = t->saved_ir_context;
    }
}

/** Realize `t` and, if present, its gradient tensor. Returns 0 on success. */
int tensor_realize_with_grads(Tensor* t) {
    if (!t)
        return -1;
    int rc = tensor_realize(t);
    if (rc != 0)
        return rc;
    if (t->grad)
        rc = tensor_realize(t->grad);
    return rc;
}

/** Alias for tensor_realize; present for API symmetry with lazy frameworks. */
int tensor_schedule(Tensor* t) { return tensor_realize(t); }

/** Block until `t`'s device work completes. No-op on the CPU backend. */
int tensor_sync(Tensor* t) {
    (void)t;
    return 0;
}
