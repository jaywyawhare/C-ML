

#ifndef CML_SHAPE_TRACKER_H
#define CML_SHAPE_TRACKER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct STView {
    int*     shape;        
    int64_t* strides;      
    int64_t  offset;       
    
    int64_t* mask_begin;   
    int64_t* mask_end;     
    bool     has_mask;
    int      ndim;
} STView;

typedef struct ShapeTracker {
    STView** views;         
    int      num_views;
    int      views_capacity;
} ShapeTracker;

STView* st_view_create(const int* shape, const int64_t* strides, int64_t offset,
                       const int64_t* mask_begin, const int64_t* mask_end, int ndim);
STView* st_view_copy(const STView* v);
void    st_view_free(STView* v);

STView* st_view_from_shape(const int* shape, int ndim);

bool st_view_is_contiguous(const STView* v);

ShapeTracker* shape_tracker_create(const int* shape, int ndim);
ShapeTracker* shape_tracker_copy(const ShapeTracker* st);
void          shape_tracker_free(ShapeTracker* st);

const int* shape_tracker_shape(const ShapeTracker* st);
int        shape_tracker_ndim(const ShapeTracker* st);
int64_t    shape_tracker_numel(const ShapeTracker* st);

int shape_tracker_reshape(ShapeTracker* st, const int* new_shape, int new_ndim);

int shape_tracker_permute(ShapeTracker* st, const int* perm);

int shape_tracker_expand(ShapeTracker* st, const int* new_shape, int new_ndim);

int shape_tracker_shrink(ShapeTracker* st, const int64_t* begin, const int64_t* end);

int shape_tracker_pad(ShapeTracker* st, const int64_t* before, const int64_t* after);

int shape_tracker_stride(ShapeTracker* st, const int64_t* strides);

int shape_tracker_flip(ShapeTracker* st, const bool* flip_dims);

int shape_tracker_index_expr(const ShapeTracker* st,
                              const char* const* loop_vars,
                              char* out_buf,   size_t out_size,
                              char* valid_buf, size_t valid_size);

int shape_tracker_simplify(ShapeTracker* st);

bool shape_tracker_is_contiguous(const ShapeTracker* st);

void shape_tracker_print(const ShapeTracker* st);

#ifdef __cplusplus
}
#endif

#endif 
