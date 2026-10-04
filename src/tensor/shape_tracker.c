#include "tensor/shape_tracker.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "alloc/cml_allocator.h"

/** Allocate and copy an n-element int64 array; NULL on NULL input or OOM. */
static int64_t* dup_i64(const int64_t* src, int n) {
    if (!src)
        return NULL;
    int64_t* dst = cml_malloc((size_t)n * sizeof(int64_t));
    if (dst)
        memcpy(dst, src, (size_t)n * sizeof(int64_t));
    return dst;
}

/** Allocate and copy an n-element int array; NULL on NULL input or OOM. */
static int* dup_int(const int* src, int n) {
    if (!src)
        return NULL;
    int* dst = cml_malloc((size_t)n * sizeof(int));
    if (dst)
        memcpy(dst, src, (size_t)n * sizeof(int));
    return dst;
}

/** Allocate a view owning copies of shape/strides and, if both are given, the
 *  begin/end mask. Returns NULL on bad args or OOM (partial state is freed). */
STView* st_view_create(const int* shape, const int64_t* strides, int64_t offset,
                       const int64_t* mask_begin, const int64_t* mask_end, int ndim) {
    if (!shape || ndim <= 0)
        return NULL;
    STView* v = cml_calloc(1, sizeof(STView));
    if (!v)
        return NULL;
    v->ndim    = ndim;
    v->offset  = offset;
    v->shape   = dup_int(shape, ndim);
    v->strides = dup_i64(strides, ndim);
    if (!v->shape || !v->strides)
        goto fail;
    if (mask_begin && mask_end) {
        v->mask_begin = dup_i64(mask_begin, ndim);
        v->mask_end   = dup_i64(mask_end, ndim);
        if (!v->mask_begin || !v->mask_end)
            goto fail;
        v->has_mask = true;
    }
    return v;
fail:
    st_view_free(v);
    return NULL;
}

/** Deep-copy a view (shape, strides, offset, mask). NULL on NULL input or OOM. */
STView* st_view_copy(const STView* v) {
    if (!v)
        return NULL;
    return st_view_create(v->shape, v->strides, v->offset, v->mask_begin, v->mask_end, v->ndim);
}

/** Free a view and all arrays it owns. */
void st_view_free(STView* v) {
    if (!v)
        return;
    cml_free(v->shape);
    cml_free(v->strides);
    cml_free(v->mask_begin);
    cml_free(v->mask_end);
    cml_free(v);
}

/** Build a contiguous, unmasked view for `shape`; size-1 dims get stride 0 (broadcastable). */
STView* st_view_from_shape(const int* shape, int ndim) {
    if (!shape || ndim <= 0)
        return NULL;
    int64_t* strides = cml_malloc((size_t)ndim * sizeof(int64_t));
    if (!strides)
        return NULL;

    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = (shape[i] == 1) ? 0 : stride;
        stride *= shape[i];
    }
    STView* v = st_view_create(shape, strides, 0, NULL, NULL, ndim);
    cml_free(strides);
    return v;
}

/** True if the view is dense row-major with zero offset and no mask (stride-0
 *  broadcast dims allowed). */
bool st_view_is_contiguous(const STView* v) {
    if (!v || v->has_mask || v->offset != 0)
        return false;
    int64_t expected = 1;
    for (int i = v->ndim - 1; i >= 0; --i) {
        if (v->strides[i] != 0 && v->strides[i] != expected)
            return false;
        expected *= v->shape[i];
    }
    return true;
}

#define ST_INIT_CAPACITY 4

/** Create a tracker seeded with a single contiguous view of `shape`. NULL on OOM. */
ShapeTracker* shape_tracker_create(const int* shape, int ndim) {
    if (!shape || ndim <= 0)
        return NULL;
    ShapeTracker* st = cml_calloc(1, sizeof(ShapeTracker));
    if (!st)
        return NULL;
    st->views = cml_malloc(ST_INIT_CAPACITY * sizeof(STView*));
    if (!st->views) {
        cml_free(st);
        return NULL;
    }
    st->views_capacity = ST_INIT_CAPACITY;
    st->num_views      = 0;
    STView* v          = st_view_from_shape(shape, ndim);
    if (!v) {
        cml_free(st->views);
        cml_free(st);
        return NULL;
    }
    st->views[0]  = v;
    st->num_views = 1;
    return st;
}

/** Deep-copy a tracker and every view in its stack. NULL on OOM. */
ShapeTracker* shape_tracker_copy(const ShapeTracker* src) {
    if (!src)
        return NULL;
    ShapeTracker* dst = cml_calloc(1, sizeof(ShapeTracker));
    if (!dst)
        return NULL;
    dst->views = cml_malloc((size_t)src->num_views * sizeof(STView*));
    if (!dst->views) {
        cml_free(dst);
        return NULL;
    }
    dst->views_capacity = src->num_views;
    for (int i = 0; i < src->num_views; ++i) {
        dst->views[i] = st_view_copy(src->views[i]);
        if (!dst->views[i]) {
            for (int j = 0; j < i; ++j)
                st_view_free(dst->views[j]);
            cml_free(dst->views);
            cml_free(dst);
            return NULL;
        }
    }
    dst->num_views = src->num_views;
    return dst;
}

/** Free the tracker and all views it holds. */
void shape_tracker_free(ShapeTracker* st) {
    if (!st)
        return;
    for (int i = 0; i < st->num_views; ++i)
        st_view_free(st->views[i]);
    cml_free(st->views);
    cml_free(st);
}

/** The current (innermost) view on the stack, or NULL if empty. */
static STView* st_top(const ShapeTracker* st) {
    return (st->num_views > 0) ? st->views[st->num_views - 1] : NULL;
}

/** Push a view, growing the stack as needed. Returns 0, or -1 on OOM. */
static int st_push(ShapeTracker* st, STView* v) {
    if (st->num_views >= st->views_capacity) {
        int new_cap  = st->views_capacity * 2;
        STView** tmp = cml_realloc(st->views, (size_t)new_cap * sizeof(STView*));
        if (!tmp)
            return -1;
        st->views          = tmp;
        st->views_capacity = new_cap;
    }
    st->views[st->num_views++] = v;
    return 0;
}

/** Shape of the current view (borrowed, do not free); NULL if empty. */
const int* shape_tracker_shape(const ShapeTracker* st) {
    STView* v = st_top(st);
    return v ? v->shape : NULL;
}

/** Rank of the current view; 0 if empty. */
int shape_tracker_ndim(const ShapeTracker* st) {
    STView* v = st_top(st);
    return v ? v->ndim : 0;
}

/** Product of the current view's dimensions; 0 if empty. */
int64_t shape_tracker_numel(const ShapeTracker* st) {
    STView* v = st_top(st);
    if (!v)
        return 0;
    int64_t n = 1;
    for (int i = 0; i < v->ndim; ++i)
        n *= v->shape[i];
    return n;
}

/** Reshape the top view (numel must be preserved): rewrites it in place when
 *  contiguous, else pushes a fresh view. Returns 0 on success, -1 on error. */
int shape_tracker_reshape(ShapeTracker* st, const int* new_shape, int new_ndim) {
    if (!st || !new_shape || new_ndim <= 0)
        return -1;
    STView* top = st_top(st);
    if (!top)
        return -1;

    int64_t old_numel = 1, new_numel = 1;
    for (int i = 0; i < top->ndim; ++i)
        old_numel *= top->shape[i];
    for (int i = 0; i < new_ndim; ++i)
        new_numel *= new_shape[i];
    if (old_numel != new_numel)
        return -1;

    if (st_view_is_contiguous(top)) {
        cml_free(top->shape);
        cml_free(top->strides);
        top->shape   = dup_int(new_shape, new_ndim);
        top->strides = cml_malloc((size_t)new_ndim * sizeof(int64_t));
        if (!top->shape || !top->strides)
            return -1;
        int64_t stride = 1;
        for (int i = new_ndim - 1; i >= 0; --i) {
            top->strides[i] = (new_shape[i] == 1) ? 0 : stride;
            stride *= new_shape[i];
        }
        top->ndim = new_ndim;
        return 0;
    }

    STView* v = st_view_from_shape(new_shape, new_ndim);
    if (!v)
        return -1;
    return st_push(st, v);
}

/** Reorder the top view's axes by `perm` (and its mask, if any), editing it in
 *  place. Returns 0 on success, -1 on bad permutation or OOM. */
int shape_tracker_permute(ShapeTracker* st, const int* perm) {
    if (!st || !perm)
        return -1;
    STView* top = st_top(st);
    if (!top)
        return -1;
    int ndim = top->ndim;

    int* new_shape       = cml_malloc((size_t)ndim * sizeof(int));
    int64_t* new_strides = cml_malloc((size_t)ndim * sizeof(int64_t));
    if (!new_shape || !new_strides) {
        cml_free(new_shape);
        cml_free(new_strides);
        return -1;
    }

    for (int i = 0; i < ndim; ++i) {
        int p = perm[i];
        if (p < 0 || p >= ndim) {
            cml_free(new_shape);
            cml_free(new_strides);
            return -1;
        }
        new_shape[i]   = top->shape[p];
        new_strides[i] = top->strides[p];
    }

    memcpy(top->shape, new_shape, (size_t)ndim * sizeof(int));
    memcpy(top->strides, new_strides, (size_t)ndim * sizeof(int64_t));
    cml_free(new_shape);
    cml_free(new_strides);

    if (top->has_mask) {
        int64_t* nb = cml_malloc((size_t)ndim * sizeof(int64_t));
        int64_t* ne = cml_malloc((size_t)ndim * sizeof(int64_t));
        if (!nb || !ne) {
            cml_free(nb);
            cml_free(ne);
            return -1;
        }
        for (int i = 0; i < ndim; ++i) {
            nb[i] = top->mask_begin[perm[i]];
            ne[i] = top->mask_end[perm[i]];
        }
        memcpy(top->mask_begin, nb, (size_t)ndim * sizeof(int64_t));
        memcpy(top->mask_end, ne, (size_t)ndim * sizeof(int64_t));
        cml_free(nb);
        cml_free(ne);
    }
    return 0;
}

/** Broadcast the top view to `new_shape`: prepended and size-1 dims get stride 0.
 *  Returns 0 on success, -1 if the new rank is smaller or on OOM. */
int shape_tracker_expand(ShapeTracker* st, const int* new_shape, int new_ndim) {
    if (!st || !new_shape || new_ndim <= 0)
        return -1;
    STView* top = st_top(st);
    if (!top)
        return -1;
    if (new_ndim < top->ndim)
        return -1;

    int* ns     = cml_malloc((size_t)new_ndim * sizeof(int));
    int64_t* ss = cml_malloc((size_t)new_ndim * sizeof(int64_t));
    if (!ns || !ss) {
        cml_free(ns);
        cml_free(ss);
        return -1;
    }

    int offset = new_ndim - top->ndim;
    for (int i = 0; i < new_ndim; ++i) {
        if (i < offset) {
            ns[i] = new_shape[i];
            ss[i] = 0;
        } else {
            int old_i = i - offset;
            ns[i]     = new_shape[i];
            ss[i]     = (top->shape[old_i] == 1 && new_shape[i] != 1) ? 0 : top->strides[old_i];
        }
    }
    cml_free(top->shape);
    top->shape = ns;
    cml_free(top->strides);
    top->strides = ss;
    top->ndim    = new_ndim;
    return 0;
}

/** Slice the top view to [begin, end) per axis, folding the start into the offset.
 *  Returns 0 on success, -1 if any range is out of bounds or empty. */
int shape_tracker_shrink(ShapeTracker* st, const int64_t* begin, const int64_t* end) {
    if (!st || !begin || !end)
        return -1;
    STView* top = st_top(st);
    if (!top)
        return -1;
    int ndim = top->ndim;

    int64_t extra_offset = 0;
    for (int i = 0; i < ndim; ++i) {
        if (begin[i] < 0 || end[i] > top->shape[i] || begin[i] >= end[i])
            return -1;
        extra_offset += begin[i] * top->strides[i];
        top->shape[i] = (int)(end[i] - begin[i]);
    }
    top->offset += extra_offset;
    return 0;
}

/** Pad the top view by `before`/`after` per axis, recording the in-bounds region
 *  as a mask (allocated on first pad). Returns 0 on success, -1 on OOM. */
int shape_tracker_pad(ShapeTracker* st, const int64_t* before, const int64_t* after) {
    if (!st || !before || !after)
        return -1;
    STView* top = st_top(st);
    if (!top)
        return -1;
    int ndim = top->ndim;

    if (!top->has_mask) {
        top->mask_begin = cml_malloc((size_t)ndim * sizeof(int64_t));
        top->mask_end   = cml_malloc((size_t)ndim * sizeof(int64_t));
        if (!top->mask_begin || !top->mask_end)
            return -1;
        for (int i = 0; i < ndim; ++i) {
            top->mask_begin[i] = 0;
            top->mask_end[i]   = top->shape[i];
        }
        top->has_mask = true;
    }

    for (int i = 0; i < ndim; ++i) {
        top->mask_begin[i] += before[i];
        top->mask_end[i] += before[i];
        top->shape[i] += (int)(before[i] + after[i]);
    }

    for (int i = 0; i < ndim; ++i)
        top->offset -= before[i] * top->strides[i];
    return 0;
}

/** Subsample the top view by positive per-axis steps, scaling strides and shrinking
 *  sizes (ceil division). Returns 0 on success, -1 on a non-positive step. */
int shape_tracker_stride(ShapeTracker* st, const int64_t* new_strides) {
    if (!st || !new_strides)
        return -1;
    STView* top = st_top(st);
    if (!top)
        return -1;
    int ndim = top->ndim;

    for (int i = 0; i < ndim; ++i) {
        if (new_strides[i] <= 0)
            return -1;
        if (top->shape[i] > 1)
            top->shape[i] = (int)((top->shape[i] + new_strides[i] - 1) / new_strides[i]);
        top->strides[i] *= new_strides[i];
    }
    return 0;
}

/** Reverse the marked axes by negating their strides and advancing the offset to
 *  the last element. Returns 0 on success, -1 on bad args. */
int shape_tracker_flip(ShapeTracker* st, const bool* flip_dims) {
    if (!st || !flip_dims)
        return -1;
    STView* top = st_top(st);
    if (!top)
        return -1;
    for (int i = 0; i < top->ndim; ++i) {
        if (!flip_dims[i])
            continue;
        top->offset += top->strides[i] * (top->shape[i] - 1);
        top->strides[i] = -top->strides[i];
    }
    return 0;
}

/** Render the current view's flat-index formula over `loop_vars` into `out_buf`,
 *  and, if requested, the mask's bounds predicate into `valid_buf` ("1" when
 *  unmasked). Returns 0 on success, -1 on bad args. */
int shape_tracker_index_expr(const ShapeTracker* st, const char* const* loop_vars, char* out_buf,
                             size_t out_size, char* valid_buf, size_t valid_size) {
    if (!st || !loop_vars || !out_buf || out_size == 0)
        return -1;
    STView* v = st_top(st);
    if (!v)
        return -1;

    char tmp[4096] = {0};
    int pos        = 0;
    pos += snprintf(tmp + pos, sizeof(tmp) - (size_t)pos, "%lld", (long long)v->offset);
    for (int i = 0; i < v->ndim; ++i) {
        if (v->strides[i] == 0)
            continue;
        if (v->strides[i] == 1)
            pos += snprintf(tmp + pos, sizeof(tmp) - (size_t)pos, " + %s", loop_vars[i]);
        else
            pos += snprintf(tmp + pos, sizeof(tmp) - (size_t)pos, " + %s*%lld", loop_vars[i],
                            (long long)v->strides[i]);
    }
    snprintf(out_buf, out_size, "%s", tmp);

    if (valid_buf && valid_size > 0) {
        if (!v->has_mask) {
            snprintf(valid_buf, valid_size, "1");
        } else {
            char vtmp[4096] = {0};
            int vpos        = 0;
            bool first      = true;
            for (int i = 0; i < v->ndim; ++i) {
                if (v->mask_begin[i] == 0 && v->mask_end[i] == v->shape[i])
                    continue;
                if (!first)
                    vpos += snprintf(vtmp + vpos, sizeof(vtmp) - (size_t)vpos, " && ");
                if (v->mask_begin[i] != 0)
                    vpos += snprintf(vtmp + vpos, sizeof(vtmp) - (size_t)vpos, "%s >= %lld",
                                     loop_vars[i], (long long)v->mask_begin[i]);
                if (v->mask_begin[i] != 0 && v->mask_end[i] != v->shape[i])
                    vpos += snprintf(vtmp + vpos, sizeof(vtmp) - (size_t)vpos, " && ");
                if (v->mask_end[i] != v->shape[i])
                    vpos += snprintf(vtmp + vpos, sizeof(vtmp) - (size_t)vpos, "%s < %lld",
                                     loop_vars[i], (long long)v->mask_end[i]);
                first = false;
            }
            snprintf(valid_buf, valid_size, "%s", first ? "1" : vtmp);
        }
    }
    return 0;
}

/** Collapse stacked views where an inner contiguous view sits over an unmasked
 *  outer one. Returns the number of views removed. */
int shape_tracker_simplify(ShapeTracker* st) {
    if (!st || st->num_views < 2)
        return 0;
    int collapsed = 0;
    int i         = st->num_views - 1;
    while (i > 0) {
        STView* inner = st->views[i];
        STView* outer = st->views[i - 1];

        if (st_view_is_contiguous(inner) && !outer->has_mask) {
            st_view_free(inner);
            for (int j = i; j < st->num_views - 1; ++j)
                st->views[j] = st->views[j + 1];
            st->num_views--;
            ++collapsed;
        } else {
            --i;
        }
    }
    return collapsed;
}

/** True only if every view in the stack is contiguous. */
bool shape_tracker_is_contiguous(const ShapeTracker* st) {
    if (!st)
        return false;
    for (int i = 0; i < st->num_views; ++i)
        if (!st_view_is_contiguous(st->views[i]))
            return false;
    return true;
}

/** Dump every view (shape, strides, offset, mask) to stderr for debugging. */
void shape_tracker_print(const ShapeTracker* st) {
    if (!st) {
        fprintf(stderr, "ShapeTracker(NULL)\n");
        return;
    }
    fprintf(stderr, "ShapeTracker(%d view%s):\n", st->num_views, st->num_views == 1 ? "" : "s");
    for (int v = 0; v < st->num_views; ++v) {
        STView* view = st->views[v];
        fprintf(stderr, "  [%d] shape=(", v);
        for (int i = 0; i < view->ndim; ++i)
            fprintf(stderr, "%d%s", view->shape[i], i < view->ndim - 1 ? "," : "");
        fprintf(stderr, ") strides=(");
        for (int i = 0; i < view->ndim; ++i)
            fprintf(stderr, "%lld%s", (long long)view->strides[i], i < view->ndim - 1 ? "," : "");
        fprintf(stderr, ") offset=%lld", (long long)view->offset);
        if (view->has_mask) {
            fprintf(stderr, " mask=[");
            for (int i = 0; i < view->ndim; ++i)
                fprintf(stderr, "(%lld,%lld)%s", (long long)view->mask_begin[i],
                        (long long)view->mask_end[i], i < view->ndim - 1 ? "," : "");
            fprintf(stderr, "]");
        }
        fprintf(stderr, "\n");
    }
}
