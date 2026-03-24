

#ifndef CML_DTYPE_VEC_H
#define CML_DTYPE_VEC_H

#include "tensor/tensor.h"  
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VEC_WIDTH_NONE = 1,
    VEC_WIDTH_2    = 2,
    VEC_WIDTH_4    = 4,
    VEC_WIDTH_8    = 8,
    VEC_WIDTH_16   = 16,
} VecWidth;

typedef struct VecDType {
    DType    scalar;    
    int      n;         
} VecDType;

static inline VecDType dtype_scalar(DType d) {
    VecDType vt; vt.scalar = d; vt.n = 1; return vt;
}

VecDType dtype_vec(DType d, int n);

bool dtype_vec_valid(VecDType vt);

size_t dtype_vec_size(VecDType vt);

const char* dtype_vec_c_name(VecDType vt);

size_t dtype_vec_alignment(VecDType vt);

static inline bool dtype_vec_is_scalar(VecDType vt) { return vt.n == 1; }

VecDType dtype_vec_widen(VecDType vt);

VecDType dtype_vec_narrow(VecDType vt);

int dtype_vec_max_width(DType d);

int dtype_vec_splat(VecDType vt, double val, char* buf, size_t buf_size);

int dtype_vec_lane(VecDType vt, int lane, const char* vec_name,
                   char* buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif 
