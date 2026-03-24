/*
 * ShapeTracker — composable view stack for zero-copy tensor transformations.
 *
 * Mirrors TinyGrad's ShapeTracker: a tensor is represented as a stack of Views.
 * Each View encodes a linear index mapping:
 *   flat_index = offset + sum_i(index_i * strides[i])
 * with an optional per-dimension validity mask for pad/shrink.
 *
 * Views compose without data movement; codegen reads the index expressions
 * to emit correct memory accesses.
 */

#ifndef CML_SHAPE_TRACKER_H
#define CML_SHAPE_TRACKER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * STView — a single affine index mapping with optional mask
 * ------------------------------------------------------------------------- */

typedef struct STView {
    int*     shape;        /* output shape  [ndim]  (owned) */
    int64_t* strides;      /* stride per dim [ndim]; 0 = broadcast (owned) */
    int64_t  offset;       /* base offset in elements */
    /* mask: valid element range [mask_begin[i], mask_end[i]) per dim.
     * NULL when the whole dimension is valid (no padding). */
    int64_t* mask_begin;   /* [ndim] or NULL (owned) */
    int64_t* mask_end;     /* [ndim] or NULL (owned) */
    bool     has_mask;
    int      ndim;
} STView;

/* -------------------------------------------------------------------------
 * ShapeTracker — ordered stack of views
 * ------------------------------------------------------------------------- */

typedef struct ShapeTracker {
    STView** views;         /* views[0] = outermost, views[n-1] = innermost */
    int      num_views;
    int      views_capacity;
} ShapeTracker;

/* ---- STView lifecycle ---- */
STView* st_view_create(const int* shape, const int64_t* strides, int64_t offset,
                       const int64_t* mask_begin, const int64_t* mask_end, int ndim);
STView* st_view_copy(const STView* v);
void    st_view_free(STView* v);

/* Build a contiguous view for a concrete shape (row-major strides). */
STView* st_view_from_shape(const int* shape, int ndim);

/* True when the view has no mask and strides match row-major order. */
bool st_view_is_contiguous(const STView* v);

/* ---- ShapeTracker lifecycle ---- */
ShapeTracker* shape_tracker_create(const int* shape, int ndim);
ShapeTracker* shape_tracker_copy(const ShapeTracker* st);
void          shape_tracker_free(ShapeTracker* st);

/* ---- Current logical shape (top view) ---- */
const int* shape_tracker_shape(const ShapeTracker* st);
int        shape_tracker_ndim(const ShapeTracker* st);
int64_t    shape_tracker_numel(const ShapeTracker* st);

/* ---- Zero-copy movement ops (each pushes or modifies the top view) ---- */

/* Reshape: add a new view with row-major strides for new_shape.
 * Collapses current top view if possible (no mask, contiguous). */
int shape_tracker_reshape(ShapeTracker* st, const int* new_shape, int new_ndim);

/* Permute dimensions in-place on the top view. */
int shape_tracker_permute(ShapeTracker* st, const int* perm);

/* Expand (broadcast): set strides to 0 for size-1 or new dims. */
int shape_tracker_expand(ShapeTracker* st, const int* new_shape, int new_ndim);

/* Shrink: narrow each dim to [begin[i], end[i]) by adjusting offset+mask. */
int shape_tracker_shrink(ShapeTracker* st, const int64_t* begin, const int64_t* end);

/* Pad: extend each dim by (before[i], after[i]) elements; sets mask. */
int shape_tracker_pad(ShapeTracker* st, const int64_t* before, const int64_t* after);

/* Stride: subsample dim i with stride[i] (e.g. 2 = take every other element). */
int shape_tracker_stride(ShapeTracker* st, const int64_t* strides);

/* Flip: negate strides and adjust offsets for dim[i] where flip[i] != 0. */
int shape_tracker_flip(ShapeTracker* st, const bool* flip_dims);

/* ---- Index expression generation ---- */

/*
 * Generate the flat storage index expression for a set of loop variables.
 * loop_vars[i] is the name of the loop variable for dim i of the logical shape.
 * out_buf receives a C expression string (e.g. "3 + x0*8 + x1*1").
 * Returns 0 on success.  When has_mask is set, *valid_buf receives a boolean
 * guard expression; caller may pass NULL to skip validity.
 */
int shape_tracker_index_expr(const ShapeTracker* st,
                              const char* const* loop_vars,
                              char* out_buf,   size_t out_size,
                              char* valid_buf, size_t valid_size);

/* ---- Simplification ---- */

/* Collapse adjacent compatible views to reduce view-stack depth.
 * Returns number of views collapsed. */
int shape_tracker_simplify(ShapeTracker* st);

/* True when all views are contiguous and have no mask. */
bool shape_tracker_is_contiguous(const ShapeTracker* st);

/* Debug dump to stderr. */
void shape_tracker_print(const ShapeTracker* st);

#ifdef __cplusplus
}
#endif

#endif /* CML_SHAPE_TRACKER_H */
