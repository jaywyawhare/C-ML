/*
 * dtype_vec — vector (SIMD) data type descriptors.
 *
 * Mirrors TinyGrad's dtypes.float.vec(4) → float4 pattern.
 * A VecDType pairs a scalar DType with a vector width n ∈ {2,4,8,16}.
 * The codegen uses this to emit "float4", "half8", etc. and to select
 * vectorised load/store instructions.
 */

#ifndef CML_DTYPE_VEC_H
#define CML_DTYPE_VEC_H

#include "tensor/tensor.h"  /* for DType */
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Supported vector widths. */
typedef enum {
    VEC_WIDTH_NONE = 1,
    VEC_WIDTH_2    = 2,
    VEC_WIDTH_4    = 4,
    VEC_WIDTH_8    = 8,
    VEC_WIDTH_16   = 16,
} VecWidth;

typedef struct VecDType {
    DType    scalar;    /* underlying element type */
    int      n;         /* vector width: 1 (scalar), 2, 4, 8, or 16 */
} VecDType;

/* ---- Constructors ---- */

/* scalar: VecDType{dtype, 1} */
static inline VecDType dtype_scalar(DType d) {
    VecDType vt; vt.scalar = d; vt.n = 1; return vt;
}

/* vec: VecDType{d, n} — returns invalid width if n not in {1,2,4,8,16} */
VecDType dtype_vec(DType d, int n);

/* ---- Queries ---- */

/* Is n a valid vector width for d on the current platform? */
bool dtype_vec_valid(VecDType vt);

/* Total byte size of the vector type (n * sizeof(scalar)). */
size_t dtype_vec_size(VecDType vt);

/* C99 / CUDA / Metal / OpenCL type name: "float4", "half8", "__bf16x2", etc.
 * Returns "" for unsupported combinations. */
const char* dtype_vec_c_name(VecDType vt);

/* Alignment requirement in bytes for aligned load/store. */
size_t dtype_vec_alignment(VecDType vt);

/* Is this a pure scalar type (n == 1)? */
static inline bool dtype_vec_is_scalar(VecDType vt) { return vt.n == 1; }

/* Widen: double the vector width (float4 → float8), up to width-16. */
VecDType dtype_vec_widen(VecDType vt);

/* Halve the vector width (float4 → float2), down to scalar. */
VecDType dtype_vec_narrow(VecDType vt);

/* Maximum useful vector width for dtype d on the current platform. */
int dtype_vec_max_width(DType d);

/* ---- Conversion helpers used by codegen ---- */

/* Emit a C splat literal: "(float4)(val)" or similar.
 * Writes into buf[buf_size]. Returns 0 on success. */
int dtype_vec_splat(VecDType vt, double val, char* buf, size_t buf_size);

/* Emit element-extract: result = "v.x" / "v.s0" etc. for lane i. */
int dtype_vec_lane(VecDType vt, int lane, const char* vec_name,
                   char* buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* CML_DTYPE_VEC_H */
