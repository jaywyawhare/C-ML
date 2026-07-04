/*
 * torch_eager.h — Zero-IR eager execution for hot ops.
 *
 * When eager mode is enabled, hot tensor ops (add/sub/mul/div, matmul, relu,
 * sigmoid, tanh) and fused linear(+relu) compute IMMEDIATELY into materialized
 * leaf tensors using SIMD + BLAS — no IR node allocation, no intern hashing,
 * no graph teardown, and no context reset needed in tight inference loops.
 *
 * Eager mode does not build an autograd graph. To preserve correct gradients,
 * the fast path is skipped automatically whenever grad is enabled and an input
 * requires grad (it falls back to the lazy IR path). Use torch_inference_mode()
 * to enable eager + no_grad together for inference.
 *
 * Eager mode and BLAS thread count are process-global (not thread-local).
 * Concurrent updates from multiple threads are last-writer-wins; configure once
 * at startup in multi-threaded hosts.
 */

#ifndef CML_TORCH_EAGER_H
#define CML_TORCH_EAGER_H

#include "core/export.h"
#include "tensor/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Enable/disable zero-IR eager execution of hot ops (process-global). */
CML_API void torch_set_eager_mode(bool enabled);
CML_API bool torch_is_eager_mode(void);

/* Convenience: eager + no_grad on enable; restore prior grad state on disable
 * when the outermost nesting level exits (nested calls are depth-tracked). */
CML_API void torch_inference_mode(bool enabled);

/* GEMM backend thread tuning (MKL/OpenBLAS/BLIS/ILP64; process-global).
 * Concurrent calls race; typical usage is once at startup. */
CML_API void torch_set_num_threads(int n);
CML_API int  torch_get_num_threads(void);

/* Materialize a lazy tensor and detach it from the IR graph so it survives
 * torch_reset_ir(). Returns 0 on success, -1 if t is NULL or realize fails. */
CML_API int torch_realize(Tensor* t);

/* Fused functional ops: out = input @ weight^T (+ bias) [+ relu].
 * Always available (independent of eager-mode flag); compute immediately when
 * inputs are materializable float32 CPU tensors that do not require grad,
 * otherwise fall back to the lazy IR path. */
CML_API Tensor* torch_linear(Tensor* input, Tensor* weight, Tensor* bias);
CML_API Tensor* torch_linear_relu(Tensor* input, Tensor* weight, Tensor* bias);

/* Internal: attempt eager compute; return NULL to signal "use lazy path".
 * `uop` is a UOpType value (UOP_ADD, UOP_MUL, UOP_MATMUL, UOP_RELU, ...). */
Tensor* torch_eager_binary(int uop, Tensor* a, Tensor* b);
Tensor* torch_eager_unary(int uop, Tensor* a);

#ifdef __cplusplus
}
#endif

#endif /* CML_TORCH_EAGER_H */
