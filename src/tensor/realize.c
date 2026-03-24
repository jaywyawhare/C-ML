#include "tensor/realize.h"
#include "ops/ir/execution.h"
#include "core/logging.h"
#include <stdlib.h>

bool tensor_is_realized(const Tensor* t) {
    return t != NULL && t->data != NULL;
}

int tensor_realize(Tensor* t) {
    if (!t) return -1;
    if (tensor_is_realized(t)) return 0;
    
    return tensor_ensure_executed(t);
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
