/**
 * @file tensor_views.c
 * @brief Tensor view operations with stride tracking
 *
 * Implements efficient tensor views that share underlying data:
 * - tensor_contiguous(): Create contiguous copy if needed
 * - tensor_reshape(): View when possible, copy when necessary
 * - tensor_view(): Create view with different shape
 * - tensor_as_strided(): Create view with custom strides
 */

#include <stdlib.h>
#include <string.h>
#include "tensor/tensor.h"
#include "Core/logging.h"
#include "Core/memory_management.h"

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

    if (t->is_contiguous) {
        return tensor_clone(t);
    }

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
        Tensor* view = CM_MALLOC(sizeof(Tensor));
        if (!view) {
            LOG_ERROR("Failed to allocate memory for tensor view");
            return NULL;
        }

        view->shape = tensor_shape_copy(new_shape, new_ndim);
        if (!view->shape) {
            CM_FREE(view);
            return NULL;
        }

        view->strides = compute_contiguous_strides(new_shape, new_ndim);
        if (!view->strides) {
            CM_FREE(view->shape);
            CM_FREE(view);
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
        view->grad_fn       = NULL;
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

// Create view with custom shape (must have same numel)
Tensor* tensor_view(Tensor* t, int* new_shape, int new_ndim) {
    // tensor_view is an alias for tensor_reshape with view-only semantics
    return tensor_reshape(t, new_shape, new_ndim);
}

// Create view with custom strides (advanced, use with caution)
Tensor* tensor_as_strided(Tensor* t, int* shape, int ndim, size_t* strides, size_t storage_offset) {
    if (!t || !shape || !strides || ndim <= 0) {
        LOG_ERROR("Invalid arguments to tensor_as_strided");
        return NULL;
    }

    Tensor* view = CM_MALLOC(sizeof(Tensor));
    if (!view) {
        LOG_ERROR("Failed to allocate memory for strided view");
        return NULL;
    }

    // Copy shape
    view->shape = tensor_shape_copy(shape, ndim);
    if (!view->shape) {
        CM_FREE(view);
        return NULL;
    }

    // Copy strides
    view->strides = CM_MALLOC(ndim * sizeof(size_t));
    if (!view->strides) {
        CM_FREE(view->shape);
        CM_FREE(view);
        return NULL;
    }
    memcpy(view->strides, strides, ndim * sizeof(size_t));

    // Share data
    view->data           = t->data;
    view->ndim           = ndim;
    view->numel          = tensor_numel(shape, ndim);
    view->dtype          = t->dtype;
    view->device         = t->device;
    view->storage_offset = storage_offset;

    // Check if contiguous
    view->is_contiguous = check_is_contiguous(shape, strides, ndim);
    view->owns_data     = false;

    // Autograd
    view->grad          = NULL;
    view->grad_fn       = NULL;
    view->requires_grad = t->requires_grad;
    view->ref_count     = 1;
    view->base          = t->base ? t->base : t;

    LOG_DEBUG("Created custom strided view");

    return view;
}

// Compute linear offset from multi-dimensional indices
size_t tensor_compute_offset(Tensor* t, int* indices) {
    if (!t || !indices)
        return 0;

    size_t offset = t->storage_offset;
    for (int i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }

    return offset;
}
