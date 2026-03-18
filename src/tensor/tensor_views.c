#include <stdlib.h>
#include <string.h>
#include "tensor/tensor.h"
#include "tensor/tensor_views.h"
#include "tensor/tensor_manipulation.h"
#include "core/logging.h"

static bool can_reshape_as_view(Tensor* t, int* new_shape, int new_ndim) {
    if (!t->is_contiguous)
        return false;

    size_t new_numel = tensor_numel(new_shape, new_ndim);
    if (new_numel != t->numel)
        return false;

    return true;
}

Tensor* tensor_contiguous(Tensor* t) {
    if (!t)
        return NULL;

    return tensor_clone(t);
}

Tensor* tensor_reshape(Tensor* t, int* new_shape, int new_ndim) {
    if (!t || !new_shape || new_ndim <= 0) {
        LOG_ERROR("Invalid arguments to tensor_reshape");
        return NULL;
    }

    size_t new_numel = tensor_numel(new_shape, new_ndim);
    if (new_numel != t->numel) {
        LOG_ERROR("Cannot reshape tensor: numel mismatch (%zu vs %zu)", new_numel, t->numel);
        return NULL;
    }

    if (can_reshape_as_view(t, new_shape, new_ndim)) {
        Tensor* view = calloc(1, sizeof(Tensor));
        if (!view) {
            LOG_ERROR("Failed to allocate memory for tensor view");
            return NULL;
        }

        view->shape = tensor_shape_copy(new_shape, new_ndim);
        if (!view->shape) {
            free(view);
            return NULL;
        }

        view->strides = compute_contiguous_strides(new_shape, new_ndim);
        if (!view->strides) {
            free(view->shape);
            free(view);
            return NULL;
        }

        view->data           = t->data;
        view->ndim           = new_ndim;
        view->numel          = new_numel;
        view->dtype          = t->dtype;
        view->device         = t->device;
        view->storage_offset = t->storage_offset;

        view->is_contiguous = true;
        view->owns_data     = false;

        view->grad          = NULL;
        view->requires_grad = t->requires_grad;
        view->ref_count     = 1;
        view->base          = t->base ? t->base : t;

        LOG_DEBUG("Created reshape view: (%d, ...) -> (%d, ...)", t->ndim, new_ndim);

        return view;
    }

    LOG_DEBUG("Reshape requires copy (non-contiguous tensor)");
    Tensor* contiguous = tensor_contiguous(t);
    if (!contiguous)
        return NULL;

    Tensor* reshaped = tensor_reshape(contiguous, new_shape, new_ndim);
    tensor_free(contiguous);

    return reshaped;
}

Tensor* tensor_view(Tensor* t, int* new_shape, int new_ndim) {
    return tensor_reshape(t, new_shape, new_ndim);
}

Tensor* tensor_as_strided(Tensor* t, int* shape, int ndim, size_t* strides, size_t storage_offset) {
    if (!t || !shape || !strides || ndim <= 0) {
        LOG_ERROR("Invalid arguments to tensor_as_strided");
        return NULL;
    }

    Tensor* view = calloc(1, sizeof(Tensor));
    if (!view) {
        LOG_ERROR("Failed to allocate memory for strided view");
        return NULL;
    }

    view->shape = tensor_shape_copy(shape, ndim);
    if (!view->shape) {
        free(view);
        return NULL;
    }

    view->strides = malloc((size_t)ndim * sizeof(size_t));
    if (!view->strides) {
        free(view->shape);
        free(view);
        return NULL;
    }
    memcpy(view->strides, strides, (size_t)ndim * sizeof(size_t));

    view->data           = t->data;
    view->ndim           = ndim;
    view->numel          = tensor_numel(shape, ndim);
    view->dtype          = t->dtype;
    view->device         = t->device;
    view->storage_offset = storage_offset;

    view->is_contiguous = tensor_check_is_contiguous(shape, strides, ndim);
    view->owns_data     = false;

    view->grad          = NULL;
    view->requires_grad = t->requires_grad;
    view->ref_count     = 1;
    view->base          = t->base ? t->base : t;

    LOG_DEBUG("Created custom strided view");

    return view;
}

size_t tensor_compute_offset(Tensor* t, int* indices) {
    if (!t || !indices)
        return 0;

    size_t offset = t->storage_offset;
    for (int i = 0; i < t->ndim; i++) {
        offset += (size_t)indices[i] * t->strides[i];
    }

    return offset;
}
