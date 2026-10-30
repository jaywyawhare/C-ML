#include <stdlib.h>
#include <string.h>
#include "tensor/tensor.h"
#include "tensor/tensor_views.h"
#include "tensor/tensor_manipulation.h"
#include "autograd/forward_ops.h"
#include "ops/uops.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"

/**
 * True when a reshape to `new_shape` can alias `t`'s storage: `t` is contiguous
 * and the element count is unchanged.
 * @param t (Tensor*)
 * @param new_shape (int*)
 * @param new_ndim (int)
 * @return (bool)
 */
static bool can_reshape_as_view(Tensor* t, int* new_shape, int new_ndim) {
    if (!t->is_contiguous)
        return false;

    size_t new_numel = tensor_numel(new_shape, new_ndim);
    if (new_numel != t->numel)
        return false;

    return true;
}

/**
 * Return a contiguous copy of `t` (always a fresh clone). NULL on error.
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_contiguous(Tensor* t) {
    if (!t)
        return NULL;

    return tensor_clone(t);
}

/**
 * Reshape `t` (numel preserved). Returns a non-owning view sharing storage when
 * `t` is contiguous (realizing a lazy base first); otherwise reshapes a copy.
 * Returns NULL on a numel mismatch or OOM.
 * @param t (Tensor*)
 * @param new_shape (int*)
 * @param new_ndim (int)
 * @return (Tensor*)
 */
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

    /* A view aliases the base's data pointer. If the base is still lazy (an
     * IR node that has not been realized), its ->data is NULL and the view
     * would be born dangling with no way to realize later - views carry no
     * ir_node. Realize the base first so the view sees valid storage. */
    if (t->ir_node && (!t->is_executed || !t->data)) {
        tensor_ensure_executed(t);
    }

    if (can_reshape_as_view(t, new_shape, new_ndim)) {
        Tensor* view = cml_calloc(1, sizeof(Tensor));
        if (!view) {
            LOG_ERROR("Failed to allocate memory for tensor view");
            return NULL;
        }

        view->shape = tensor_shape_copy(new_shape, new_ndim);
        if (!view->shape) {
            cml_free(view);
            return NULL;
        }

        view->strides = compute_contiguous_strides(new_shape, new_ndim);
        if (!view->strides) {
            cml_free(view->shape);
            cml_free(view);
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
        view->storage       = NULL;
        /* Keep the base's data block alive while this view exists: link it to
         * shared storage (attached to the owning root on first view). When the
         * root is not an owned allocation (e.g. borrowed plan buffer) no
         * storage can be attached and the documented non-owning contract
         * applies - the base must outlive the view. */
        tensor_storage_share(view, t);

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
 * Alias for tensor_reshape (returns a view when the layout permits).
 * @param t (Tensor*)
 * @param new_shape (int*)
 * @param new_ndim (int)
 * @return (Tensor*)
 */
Tensor* tensor_view(Tensor* t, int* new_shape, int new_ndim) {
    return tensor_reshape(t, new_shape, new_ndim);
}

/**
 * Create a non-owning view over `t`'s storage with arbitrary `shape`, `strides`
 * and `storage_offset`; contiguity is derived from the strides. NULL on error.
 * The caller is responsible for keeping the strides within the backing storage.
 * @param t (Tensor*)
 * @param shape (int*)
 * @param ndim (int)
 * @param strides (size_t*)
 * @param storage_offset (size_t)
 * @return (Tensor*)
 */
Tensor* tensor_as_strided(Tensor* t, int* shape, int ndim, size_t* strides, size_t storage_offset) {
    if (!t || !shape || !strides || ndim <= 0) {
        LOG_ERROR("Invalid arguments to tensor_as_strided");
        return NULL;
    }

    Tensor* view = cml_calloc(1, sizeof(Tensor));
    if (!view) {
        LOG_ERROR("Failed to allocate memory for strided view");
        return NULL;
    }

    view->shape = tensor_shape_copy(shape, ndim);
    if (!view->shape) {
        cml_free(view);
        return NULL;
    }

    view->strides = cml_malloc((size_t)ndim * sizeof(size_t));
    if (!view->strides) {
        cml_free(view->shape);
        cml_free(view);
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
    view->storage       = NULL;
    /* Same shared-storage contract as the reshape view above. */
    tensor_storage_share(view, t);

    LOG_DEBUG("Created custom strided view");

    return view;
}

/**
 * Resolve a possibly-negative axis against `ndim`, returning -1 if out of range.
 * @param dim (int)
 * @param ndim (int)
 * @return (int)
 */
static int views_norm_dim(int dim, int ndim) {
    if (dim < 0)
        dim += ndim;
    return (dim >= 0 && dim < ndim) ? dim : -1;
}

/**
 * Reorder dimensions by `dims`, a permutation of 0..ndim-1 (lazy view op).
 * @param t (Tensor*)
 * @param dims (int*)
 * @param ndim (int)
 * @return (Tensor*)
 */
Tensor* tensor_permute(Tensor* t, int* dims, int ndim) {
    if (!t || !dims)
        return NULL;
    if (ndim != t->ndim) {
        LOG_ERROR("tensor_permute: dims length %d != tensor ndim %d", ndim, t->ndim);
        return NULL;
    }
    PermuteParams params = {.perm = dims, .num_dims = ndim};
    return uop_permute(t, &params);
}

/**
 * Broadcast to `new_shape` without copying (lazy); a -1 entry keeps that dim's
 * current size, dims aligned from the right (torch.Tensor.expand).
 * @param t (Tensor*)
 * @param new_shape (int*)
 * @param new_ndim (int)
 * @return (Tensor*)
 */
Tensor* tensor_expand(Tensor* t, int* new_shape, int new_ndim) {
    if (!t || !new_shape || new_ndim <= 0)
        return NULL;
    if (new_ndim < t->ndim) {
        LOG_ERROR("tensor_expand: target ndim %d < tensor ndim %d", new_ndim, t->ndim);
        return NULL;
    }

    int* resolved = cml_malloc((size_t)new_ndim * sizeof(int));
    if (!resolved)
        return NULL;
    int offset = new_ndim - t->ndim;
    for (int i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1) {
            if (i < offset) {
                LOG_ERROR("tensor_expand: -1 is not allowed in an expanded (new) dimension");
                cml_free(resolved);
                return NULL;
            }
            resolved[i] = t->shape[i - offset];
        } else {
            resolved[i] = new_shape[i];
        }
    }

    Tensor* result = uop_expand_to(t, resolved, new_ndim);
    cml_free(resolved);
    return result;
}

/**
 * Narrow `dim` to `length` elements starting at `start` (lazy; torch.narrow).
 * @param t (Tensor*)
 * @param dim (int)
 * @param start (int)
 * @param length (int)
 * @return (Tensor*)
 */
Tensor* tensor_narrow(Tensor* t, int dim, int start, int length) {
    if (!t)
        return NULL;
    int d = views_norm_dim(dim, t->ndim);
    if (d < 0) {
        LOG_ERROR("tensor_narrow: dim %d out of range for %dD tensor", dim, t->ndim);
        return NULL;
    }
    if (start < 0)
        start += t->shape[d];
    if (start < 0 || length < 0 || start + length > t->shape[d]) {
        LOG_ERROR("tensor_narrow: range [%d, %d) out of bounds for dim %d (size %d)", start,
                  start + length, d, t->shape[d]);
        return NULL;
    }

    int* starts = cml_malloc((size_t)t->ndim * sizeof(int));
    int* ends   = cml_malloc((size_t)t->ndim * sizeof(int));
    if (!starts || !ends) {
        cml_free(starts);
        cml_free(ends);
        return NULL;
    }
    for (int i = 0; i < t->ndim; i++) {
        starts[i] = 0;
        ends[i]   = t->shape[i];
    }
    starts[d] = start;
    ends[d]   = start + length;

    Tensor* result = uop_shrink(t, starts, ends, t->ndim);
    cml_free(starts);
    cml_free(ends);
    return result;
}

/**
 * Select slices of `t` along `dim` at the entries of the 1-D `index` tensor
 * (torch.index_select).
 * @param t (Tensor*)
 * @param dim (int)
 * @param index (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_index_select(Tensor* t, int dim, Tensor* index) {
    if (!t || !index)
        return NULL;
    int d = views_norm_dim(dim, t->ndim);
    if (d < 0) {
        LOG_ERROR("tensor_index_select: dim %d out of range for %dD tensor", dim, t->ndim);
        return NULL;
    }

    if (d == 0)
        return uop_gather(t, index, 0);

    /* uop_gather selects along axis 0, so rotate `d` to the front and back. */
    int nd   = t->ndim;
    int* fp  = cml_malloc((size_t)nd * sizeof(int));
    int* inv = cml_malloc((size_t)nd * sizeof(int));
    if (!fp || !inv) {
        cml_free(fp);
        cml_free(inv);
        return NULL;
    }
    fp[0] = d;
    for (int i = 0, j = 1; i < nd; i++)
        if (i != d)
            fp[j++] = i;
    for (int p = 0; p < nd; p++)
        inv[p] = (p < d) ? p + 1 : (p == d ? 0 : p);

    Tensor* permuted = tensor_permute(t, fp, nd);
    Tensor* gathered = permuted ? uop_gather(permuted, index, 0) : NULL;
    Tensor* result   = gathered ? tensor_permute(gathered, inv, nd) : NULL;

    if (permuted)
        tensor_free(permuted);
    if (gathered)
        tensor_free(gathered);
    cml_free(fp);
    cml_free(inv);
    return result;
}

/**
 * Sliding windows of `size` along `dim` with the given `step`, appending a new
 * last axis of length `size` (torch.Tensor.unfold).
 * @param t (Tensor*)
 * @param dim (int)
 * @param size (int)
 * @param step (int)
 * @return (Tensor*)
 */
Tensor* tensor_unfold(Tensor* t, int dim, int size, int step) {
    if (!t)
        return NULL;
    int d = views_norm_dim(dim, t->ndim);
    if (d < 0) {
        LOG_ERROR("tensor_unfold: dim %d out of range for %dD tensor", dim, t->ndim);
        return NULL;
    }

    if (d == t->ndim - 1)
        return uop_unfold(t, size, step);

    /* uop_unfold windows the last axis, so rotate `d` to the end and back. */
    int nd  = t->ndim;
    int* fp = cml_malloc((size_t)nd * sizeof(int));
    int* tp = cml_malloc((size_t)(nd + 1) * sizeof(int));
    if (!fp || !tp) {
        cml_free(fp);
        cml_free(tp);
        return NULL;
    }
    for (int i = 0, j = 0; i < nd; i++)
        if (i != d)
            fp[j++] = i;
    fp[nd - 1] = d;
    for (int p = 0; p < nd; p++)
        tp[p] = (p < d) ? p : (p == d ? nd - 1 : p - 1);
    tp[nd] = nd;

    Tensor* permuted = tensor_permute(t, fp, nd);
    Tensor* unfolded = permuted ? uop_unfold(permuted, size, step) : NULL;
    Tensor* result   = unfolded ? tensor_permute(unfolded, tp, nd + 1) : NULL;

    if (permuted)
        tensor_free(permuted);
    if (unfolded)
        tensor_free(unfolded);
    cml_free(fp);
    cml_free(tp);
    return result;
}

/**
 * Move the axis at `source` to `destination`, shifting the others (torch.movedim).
 * @param t (Tensor*)
 * @param source (int)
 * @param destination (int)
 * @return (Tensor*)
 */
Tensor* tensor_movedim(Tensor* t, int source, int destination) {
    if (!t)
        return NULL;
    int s   = views_norm_dim(source, t->ndim);
    int dst = views_norm_dim(destination, t->ndim);
    if (s < 0 || dst < 0) {
        LOG_ERROR("tensor_movedim: axis (%d -> %d) out of range for %dD tensor", source,
                  destination, t->ndim);
        return NULL;
    }

    int nd    = t->ndim;
    int* perm = cml_malloc((size_t)nd * sizeof(int));
    if (!perm)
        return NULL;
    int j = 0;
    for (int p = 0; p < nd; p++) {
        if (p == dst)
            perm[p] = s;
        else {
            if (j == s)
                j++;
            perm[p] = j++;
        }
    }

    Tensor* result = tensor_permute(t, perm, nd);
    cml_free(perm);
    return result;
}

/**
 * Swap two axes (torch.swapaxes / numpy.swapaxes).
 * @param t (Tensor*)
 * @param axis0 (int)
 * @param axis1 (int)
 * @return (Tensor*)
 */
Tensor* tensor_swapaxes(Tensor* t, int axis0, int axis1) {
    if (!t)
        return NULL;
    int a0 = views_norm_dim(axis0, t->ndim);
    int a1 = views_norm_dim(axis1, t->ndim);
    if (a0 < 0 || a1 < 0) {
        LOG_ERROR("tensor_swapaxes: axis (%d, %d) out of range for %dD tensor", axis0, axis1,
                  t->ndim);
        return NULL;
    }
    /* Normalize first: tensor_transpose maps any negative axis to -2/-1. */
    return tensor_transpose(t, a0, a1);
}

/**
 * Flat storage offset of a multi-index, honoring strides and storage_offset.
 * @param t (Tensor*)
 * @param indices (int*)
 * @return (size_t)
 */
size_t tensor_compute_offset(Tensor* t, int* indices) {
    if (!t || !indices)
        return 0;

    size_t offset = t->storage_offset;
    for (int i = 0; i < t->ndim; i++) {
        offset += (size_t)indices[i] * t->strides[i];
    }

    return offset;
}
