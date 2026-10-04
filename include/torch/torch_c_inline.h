/*
 * torch_c_inline.h - Header inlines for zero-overhead hot paths.
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

/** Fast Tensor.retain: bump ref_count with no null check (t must be non-NULL). */
static inline void torch_tensor_retain_fast(Tensor* t) { t->ref_count++; }

/** Fast Tensor.dim() via direct field access. */
static inline int torch_tensor_ndim_fast(const Tensor* t) { return t->ndim; }

/** Fast Tensor.numel() via direct field access. */
static inline size_t torch_tensor_numel_fast(const Tensor* t) { return t->numel; }

/** Fast Tensor.dtype via direct field access. */
static inline DType torch_tensor_dtype_fast(const Tensor* t) { return t->dtype; }

/** Fast Tensor.device via direct field access. */
static inline DeviceType torch_tensor_device_fast(const Tensor* t) { return t->device; }

/** Fast Tensor.is_contiguous() via direct field access. */
static inline bool torch_tensor_is_contiguous_fast(const Tensor* t) { return t->is_contiguous; }

/** Fast Tensor.requires_grad via direct field access. */
static inline bool torch_tensor_requires_grad_fast(const Tensor* t) { return t->requires_grad; }

/** Fast Tensor.size(): direct pointer to the shape array. */
static inline const int* torch_tensor_sizes_fast(const Tensor* t) { return t->shape; }

/** Fast Tensor.data_ptr() with no realize/error handling. */
static inline void* torch_tensor_data_ptr_fast(Tensor* t) { return tensor_data_ptr(t); }

/* --- Hot tensor ops (skip cml_* wrapper layer) --- */

/** torch.add() hot path skipping the cml_* wrapper layer. */
static inline Tensor* torch_add_fast(Tensor* a, Tensor* b) { return tensor_add(a, b); }
/** torch.mul() hot path skipping the cml_* wrapper layer. */
static inline Tensor* torch_mul_fast(Tensor* a, Tensor* b) { return tensor_mul(a, b); }
/** torch.matmul() hot path skipping the cml_* wrapper layer. */
static inline Tensor* torch_matmul_fast(Tensor* a, Tensor* b) { return tensor_matmul(a, b); }
/** torch.relu() hot path skipping the cml_* wrapper layer. */
static inline Tensor* torch_relu_fast(Tensor* a) { return tensor_relu(a); }

#ifdef __cplusplus
}
#endif

#endif /* CML_TORCH_C_INLINE_H */
