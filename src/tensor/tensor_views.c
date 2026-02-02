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
        Tensor* view = malloc(sizeof(Tensor));
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

/**
 * @brief Reshape tensor in-place if possible
 *
 * Attempts to reshape the tensor in-place. This only works if:
 * - The tensor is contiguous
 * - The new shape has the same number of elements
 * - The tensor owns its data (not a view)
 *
 * @param t Tensor to reshape
 * @param new_shape New shape array
 * @param new_ndim Number of dimensions in new shape
 * @return 0 on success, -1 on failure (will fail via error handler)
 */
int tensor_reshape_inplace(Tensor* t, int* new_shape, int new_ndim) {
    if (!t || !new_shape || new_ndim <= 0) {
        LOG_ERROR("Invalid arguments to tensor_reshape_inplace");
        return -1;
    }

    size_t new_numel = tensor_numel(new_shape, new_ndim);
    if (new_numel != t->numel) {
        LOG_ERROR("Cannot reshape tensor in-place: numel mismatch (%zu vs %zu). "
                  "Use tensor_reshape() to create a new tensor.",
                  new_numel, t->numel);
        return -1;
    }

    if (!t->is_contiguous) {
        LOG_ERROR("Cannot reshape non-contiguous tensor in-place. "
                  "Use tensor_reshape() to create a contiguous copy first.");
        return -1;
    }

    if (!t->owns_data) {
        LOG_ERROR("Cannot reshape tensor view in-place. "
                  "Use tensor_reshape() to create a new tensor.");
        return -1;
    }

    // Update shape
    if (t->ndim != new_ndim) {
        free(t->shape);
        t->shape = tensor_shape_copy(new_shape, new_ndim);
        if (!t->shape) {
            LOG_ERROR("Failed to allocate memory for new shape");
            return -1;
        }
    } else {
        memcpy(t->shape, new_shape, (size_t)new_ndim * sizeof(int));
    }

    // Update strides
    if (t->strides) {
        free(t->strides);
    }
    t->strides = compute_contiguous_strides(new_shape, new_ndim);
    if (!t->strides) {
        LOG_ERROR("Failed to compute strides for new shape");
        return -1;
    }

    t->ndim          = new_ndim;
    t->is_contiguous = true;

    LOG_DEBUG("Reshaped tensor in-place: (%d, ...) -> (%d, ...)", t->ndim, new_ndim);

    return 0;
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

    Tensor* view = malloc(sizeof(Tensor));
    if (!view) {
        LOG_ERROR("Failed to allocate memory for strided view");
        return NULL;
    }

    // Copy shape
    view->shape = tensor_shape_copy(shape, ndim);
    if (!view->shape) {
        free(view);
        return NULL;
    }

    // Copy strides
    view->strides = malloc((size_t)ndim * sizeof(size_t));
    if (!view->strides) {
        free(view->shape);
        free(view);
        return NULL;
    }
    memcpy(view->strides, strides, (size_t)ndim * sizeof(size_t));

    // Share data
    view->data           = t->data;
    view->ndim           = ndim;
    view->numel          = tensor_numel(shape, ndim);
    view->dtype          = t->dtype;
    view->device         = t->device;
    view->storage_offset = storage_offset;

    // Check if contiguous
    view->is_contiguous = tensor_check_is_contiguous(shape, strides, ndim);
    view->owns_data     = false;

    // Autograd
    view->grad          = NULL;
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
        offset += (size_t)indices[i] * t->strides[i];
    }

    return offset;
}
