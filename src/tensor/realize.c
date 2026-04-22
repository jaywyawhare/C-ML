#include "tensor/realize.h"
#include "ops/ir/execution.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>

bool tensor_is_realized(const Tensor* t) {
    return t != NULL && t->data != NULL;
}

int tensor_realize(Tensor* t) {
    if (!t) return -1;

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
        void* owned = malloc(nbytes);
        if (owned) {
            memcpy(owned, t->data, nbytes);
            t->data      = owned;
            t->owns_data = true;
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

int tensor_realize_all(Tensor** tensors, int num_tensors) {
    if (!tensors || num_tensors <= 0) return -1;
    int rc = 0;
    for (int i = 0; i < num_tensors; ++i) {
        if (!tensors[i] || tensor_is_realized(tensors[i])) continue;
        int r = tensor_ensure_executed(tensors[i]);
        if (r != 0) rc = r;
    }
    return rc;
}

void tensor_unrealize(Tensor* t) {
    if (!t || !t->data) return;
    if (t->owns_data) free(t->data);
    t->data        = NULL;
    t->is_executed = false;
    t->owns_data   = false;

    /* Reconnect to the IR graph so this tensor can be re-materialized.
     * Required for gradient checkpointing: free activations during the
     * forward pass, recompute on demand during backward. */
    if (t->saved_ir_node && !t->ir_node) {
        t->saved_ir_node->output    = t;
        t->saved_ir_node->is_executed = false;
        t->ir_node    = t->saved_ir_node;
        t->ir_context = t->saved_ir_context;
    }
}

int tensor_realize_with_grads(Tensor* t) {
    if (!t) return -1;
    int rc = tensor_realize(t);
    if (rc != 0) return rc;
    if (t->grad) rc = tensor_realize(t->grad);
    return rc;
}

int tensor_schedule(Tensor* t) {
    return tensor_realize(t);
}

int tensor_sync(Tensor* t) {
    (void)t;
    return 0;
}
