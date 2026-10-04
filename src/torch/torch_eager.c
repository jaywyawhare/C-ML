/*
 * torch_eager.c — Zero-IR eager execution for hot ops.
 *
 * Bypasses IR construction/teardown entirely: hot ops compute directly into
 * freshly allocated, materialized leaf tensors (tensor_create) using the SIMD
 * kernels and BLAS GEMM. This removes the ~8-12 heap allocations + string
 * copies + intern hashing each lazy op pays, and removes the need for any
 * IR context reset in inference loops (no IR accumulates).
 */

#include "torch/torch_eager.h"
#include "torch/torch_c.h"

#include "tensor/tensor.h"
#include "tensor/realize.h"
#include "autograd/autograd.h"
#include "autograd/forward_ops.h"
#include "backend/blas.h"
#include "ops/uops.h"
#include "ops/simd_math.h"

#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define TORCH_EAGER_MAX_DIMS 8

#include <stdatomic.h>

static _Atomic bool g_torch_eager        = false;
static _Atomic int g_inference_depth     = 0;
static _Atomic bool g_saved_grad_enabled = true;

/** Enable or disable process-global zero-IR eager execution of hot ops. */
void torch_set_eager_mode(bool enabled) {
    atomic_store_explicit(&g_torch_eager, enabled, memory_order_relaxed);
}

/** Query whether zero-IR eager execution is currently enabled. */
bool torch_is_eager_mode(void) {
    return atomic_load_explicit(&g_torch_eager, memory_order_relaxed);
}

/** torch.inference_mode(): depth-tracked toggle of eager + no_grad, restoring the
 *  prior grad state when the outermost scope exits. */
void torch_inference_mode(bool enabled) {
    if (enabled) {
        int prev = atomic_fetch_add_explicit(&g_inference_depth, 1, memory_order_acq_rel);
        if (prev == 0) {
            atomic_store_explicit(&g_saved_grad_enabled, torch_is_grad_enabled(),
                                  memory_order_relaxed);
            atomic_store_explicit(&g_torch_eager, true, memory_order_relaxed);
            torch_no_grad();
        }
    } else {
        int expected = atomic_load_explicit(&g_inference_depth, memory_order_acquire);
        do {
            if (expected <= 0)
                return;
        } while (!atomic_compare_exchange_weak_explicit(&g_inference_depth, &expected, expected - 1,
                                                        memory_order_acq_rel,
                                                        memory_order_acquire));
        int remaining = expected - 1;
        if (remaining == 0) {
            atomic_store_explicit(&g_torch_eager, false, memory_order_relaxed);
            if (atomic_load_explicit(&g_saved_grad_enabled, memory_order_relaxed))
                torch_enable_grad();
            else
                torch_no_grad();
        }
    }
}

/** torch.set_num_threads(): set the GEMM backend thread count (process-global). */
void torch_set_num_threads(int n) { cml_blas_set_num_threads(n); }
/** torch.get_num_threads(): current GEMM backend thread count. */
int torch_get_num_threads(void) { return cml_blas_get_num_threads(); }

/** Materialize a lazy tensor, detaching it from the IR graph; 0 on success, -1 on error. */
int torch_realize(Tensor* t) {
    if (!t)
        return -1;
    return tensor_realize(t);
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/** True if a tensor is a float32 CPU tensor eligible for the eager fast path. */
static inline bool eager_cpu_f32(const Tensor* t) {
    return t && t->dtype == DTYPE_FLOAT32 && (t->device == DEVICE_CPU || t->device == DEVICE_AUTO);
}

/** True unless an input still needs autograd tracking (then eager must be skipped). */
static inline bool eager_grad_safe(Tensor* const* ins, int n) {
    if (!torch_is_grad_enabled())
        return true;
    for (int i = 0; i < n; i++)
        if (ins[i] && ins[i]->requires_grad)
            return false;
    return true;
}

/** True if two tensors have identical rank and dimensions. */
static inline bool same_shape(const Tensor* a, const Tensor* b) {
    if (a->ndim != b->ndim)
        return false;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i])
            return false;
    return true;
}

/** Allocate a fresh materialized float32 output tensor shaped like `ref`. */
static Tensor* eager_alloc_like(const Tensor* ref) {
    return tensor_create(DTYPE_FLOAT32, ref->device, ref->ndim, ref->shape, false);
}

/** In-place ReLU over a float buffer (auto-vectorizes; no inline asm). */
static inline void relu_inplace(float* x, size_t n) {
    /* Auto-vectorizes to vmaxps under -O3 -march=native (no inline asm). */
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        x[i]    = v > 0.0f ? v : 0.0f;
    }
}

/* ------------------------------------------------------------------ */
/* Eager binary / unary                                                */
/* ------------------------------------------------------------------ */

/** Attempt eager compute of a binary op (matmul via BLAS, elementwise via SIMD);
 *  returns NULL to signal the caller should fall back to the lazy IR path. */
Tensor* torch_eager_binary(int uop, Tensor* a, Tensor* b) {
    if (!g_torch_eager || !a || !b)
        return NULL;
    Tensor* ins[2] = {a, b};
    if (!eager_grad_safe(ins, 2))
        return NULL;
    if (!eager_cpu_f32(a) || !eager_cpu_f32(b))
        return NULL;

    if (uop == UOP_MATMUL) {
        /* a: [.., M, K] (collapse leading dims into rows), b: [K, N] (2D). */
        if (a->ndim < 2 || b->ndim != 2 || a->ndim > TORCH_EAGER_MAX_DIMS)
            return NULL;
        int K = a->shape[a->ndim - 1];
        int N = b->shape[1];
        if (K <= 0 || b->shape[0] != K)
            return NULL;
        size_t rows = a->numel / (size_t)K;
        if (rows == 0 || rows > (size_t)INT_MAX)
            return NULL;

        int out_ndim = a->ndim;
        int out_shape[TORCH_EAGER_MAX_DIMS];
        for (int i = 0; i < a->ndim - 1; i++)
            out_shape[i] = a->shape[i];
        out_shape[out_ndim - 1] = N;

        float* ad = (float*)tensor_data_ptr(a);
        float* bd = (float*)tensor_data_ptr(b);
        if (!ad || !bd)
            return NULL;

        Tensor* out = tensor_create(DTYPE_FLOAT32, a->device, out_ndim, out_shape, false);
        if (!out)
            return NULL;

        CMLBlasContext* blas = cml_blas_get_context();
        if (blas && blas->initialized) {
            if (cml_blas_sgemm(blas, ad, bd, (float*)out->data, (int)rows, N, K, 1.0f, 0.0f) == 0)
                return out;
        }
        tensor_free(out);
        return NULL; /* let lazy path handle (no BLAS) */
    }

    /* Elementwise: only exact-shape match here; broadcasting → lazy path. */
    if (!same_shape(a, b))
        return NULL;

    float* ad = (float*)tensor_data_ptr(a);
    float* bd = (float*)tensor_data_ptr(b);
    if (!ad || !bd)
        return NULL;

    Tensor* out = eager_alloc_like(a);
    if (!out)
        return NULL;
    float* od = (float*)out->data;
    size_t n  = out->numel;

    switch (uop) {
    case UOP_ADD:
        simd_add_f32(ad, bd, od, n);
        break;
    case UOP_SUB:
        simd_sub_f32(ad, bd, od, n);
        break;
    case UOP_MUL:
        simd_mul_f32(ad, bd, od, n);
        break;
    case UOP_DIV:
        simd_div_f32(ad, bd, od, n);
        break;
    default:
        tensor_free(out);
        return NULL;
    }
    return out;
}

/** Attempt eager compute of a unary op (relu/sigmoid/tanh); NULL to use the lazy path. */
Tensor* torch_eager_unary(int uop, Tensor* a) {
    if (!g_torch_eager || !a)
        return NULL;
    if (!eager_grad_safe(&a, 1))
        return NULL;
    if (!eager_cpu_f32(a))
        return NULL;

    float* ad = (float*)tensor_data_ptr(a);
    if (!ad)
        return NULL;

    Tensor* out = eager_alloc_like(a);
    if (!out)
        return NULL;
    float* od = (float*)out->data;
    size_t n  = out->numel;

    switch (uop) {
    case UOP_RELU:
        memcpy(od, ad, n * sizeof(float));
        relu_inplace(od, n);
        break;
    case UOP_SIGMOID:
        simd_sigmoid_f32(ad, od, n);
        break;
    case UOP_TANH:
        simd_tanh_f32(ad, od, n);
        break;
    default:
        tensor_free(out);
        return NULL;
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* Fused linear (matmul + bias [+ relu]) — single GEMM + one epilogue  */
/* ------------------------------------------------------------------ */

/** Eager fused linear: one BLAS GEMM (input @ weight^T) plus a fused bias/relu
 *  epilogue; NULL if inputs are ineligible so the caller uses the lazy path. */
static Tensor* eager_linear(Tensor* input, Tensor* weight, Tensor* bias, bool fuse_relu) {
    if (!input || !weight)
        return NULL;
    Tensor* ins[3] = {input, weight, bias};
    if (!eager_grad_safe(ins, bias ? 3 : 2))
        return NULL;
    if (!eager_cpu_f32(input) || !eager_cpu_f32(weight) || (bias && !eager_cpu_f32(bias)))
        return NULL;
    if (input->ndim < 1 || weight->ndim != 2 || input->ndim > TORCH_EAGER_MAX_DIMS)
        return NULL;

    int K = input->shape[input->ndim - 1]; /* in_features  */
    int N = weight->shape[0];              /* out_features */
    if (K <= 0 || weight->shape[1] != K)
        return NULL;
    if (bias && bias->numel != (size_t)N)
        return NULL;
    size_t rows = input->numel / (size_t)K;
    if (rows == 0 || rows > (size_t)INT_MAX)
        return NULL;

    int out_ndim = input->ndim;
    int out_shape[TORCH_EAGER_MAX_DIMS];
    for (int i = 0; i < input->ndim - 1; i++)
        out_shape[i] = input->shape[i];
    out_shape[out_ndim - 1] = N;

    float* xd = (float*)tensor_data_ptr(input);
    float* wd = (float*)tensor_data_ptr(weight);
    float* bd = bias ? (float*)tensor_data_ptr(bias) : NULL;
    if (!xd || !wd || (bias && !bd))
        return NULL;

    CMLBlasContext* blas = cml_blas_get_context();
    if (!blas || !blas->initialized)
        return NULL;

    Tensor* out = tensor_create(DTYPE_FLOAT32, input->device, out_ndim, out_shape, false);
    if (!out)
        return NULL;
    float* od = (float*)out->data;

    /* out = input[rows,K] @ weight[N,K]^T  (single BLAS GEMM) */
    if (cml_blas_sgemm_ex(blas, xd, wd, od, (int)rows, N, K, 1.0f, 0.0f, false, true) != 0) {
        tensor_free(out);
        return NULL;
    }

    /* Fused bias + relu epilogue: one pass over the output. */
    if (bd || fuse_relu) {
        for (size_t r = 0; r < rows; r++) {
            float* o = od + r * (size_t)N;
            if (bd && fuse_relu) {
                for (int j = 0; j < N; j++) {
                    float v = o[j] + bd[j];
                    o[j]    = v > 0.0f ? v : 0.0f;
                }
            } else if (bd) {
                for (int j = 0; j < N; j++)
                    o[j] += bd[j];
            } else { /* fuse_relu only */
                for (int j = 0; j < N; j++) {
                    float v = o[j];
                    o[j]    = v > 0.0f ? v : 0.0f;
                }
            }
        }
    }
    return out;
}

/** Lazy linear fallback: build matmul(+transpose) + add + relu in the IR graph. */
static Tensor* lazy_linear(Tensor* input, Tensor* weight, Tensor* bias, bool fuse_relu) {
    if (!input || !weight)
        return NULL;
    Tensor* wt = tensor_transpose(weight, weight->ndim - 2, weight->ndim - 1);
    if (!wt)
        return NULL;
    Tensor* out = tensor_matmul(input, wt);
    if (out && bias)
        out = tensor_add(out, bias);
    if (out && fuse_relu)
        out = tensor_relu(out);
    return out;
}

/** nn.functional.linear(): out = input @ weight^T (+ bias), eager with lazy fallback. */
Tensor* torch_linear(Tensor* input, Tensor* weight, Tensor* bias) {
    Tensor* e = eager_linear(input, weight, bias, false);
    return e ? e : lazy_linear(input, weight, bias, false);
}

/** Fused linear + ReLU: out = relu(input @ weight^T + bias), eager with lazy fallback. */
Tensor* torch_linear_relu(Tensor* input, Tensor* weight, Tensor* bias) {
    Tensor* e = eager_linear(input, weight, bias, true);
    return e ? e : lazy_linear(input, weight, bias, true);
}
