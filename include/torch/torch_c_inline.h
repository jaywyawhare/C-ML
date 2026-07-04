/*
 * torch_c_inline.h — Header inlines for zero-overhead hot paths.
 *
 * Include after torch/torch_c.h. These compile to direct field access / calls
 * with no extra function-call indirection through libcml.
 *
 * *_fast helpers require non-null tensor pointers; passing NULL is undefined
 * behavior. Use the non-_fast CML_API functions for defensive null checks.
 *
 * torch_tensor_retain_fast uses plain ref_count increments to stay consistent
 * with the rest of the tensor lifecycle (not thread-safe; matches tensor.c).
 */

#ifndef CML_TORCH_C_INLINE_H
#define CML_TORCH_C_INLINE_H

#include "torch/torch_c.h"
#include "autograd/autograd.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_views.h"
#include "ops/uops.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --- Accessors (direct struct field access) --- */

static inline void torch_tensor_retain_fast(Tensor* t) { t->ref_count++; }

static inline int torch_tensor_ndim_fast(const Tensor* t) { return t->ndim; }

static inline size_t torch_tensor_numel_fast(const Tensor* t) { return t->numel; }

static inline DType torch_tensor_dtype_fast(const Tensor* t) { return t->dtype; }

static inline DeviceType torch_tensor_device_fast(const Tensor* t) { return t->device; }

static inline bool torch_tensor_is_contiguous_fast(const Tensor* t) { return t->is_contiguous; }

static inline bool torch_tensor_requires_grad_fast(const Tensor* t) { return t->requires_grad; }

static inline const int* torch_tensor_sizes_fast(const Tensor* t) { return t->shape; }

static inline void* torch_tensor_data_ptr_fast(Tensor* t) { return tensor_data_ptr(t); }

/* --- Hot tensor ops (skip cml_* wrapper layer) --- */

static inline Tensor* torch_add_fast(Tensor* a, Tensor* b) { return tensor_add(a, b); }
static inline Tensor* torch_mul_fast(Tensor* a, Tensor* b) { return tensor_mul(a, b); }
static inline Tensor* torch_matmul_fast(Tensor* a, Tensor* b) { return tensor_matmul(a, b); }
static inline Tensor* torch_relu_fast(Tensor* a) { return tensor_relu(a); }

#ifdef __cplusplus
}
#endif

#endif /* CML_TORCH_C_INLINE_H */
