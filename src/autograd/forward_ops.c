#include "autograd/autograd.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "core/error_stack.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "alloc/cml_allocator.h"

/* Same-shape unary op recorded into the lazy graph. */
static Tensor* forward_unary(Tensor* a, UOpType type) {
    if (!a)
        return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, type, inputs, 1, NULL) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape  = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim   = a->ndim;
    if (a->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
    }

    return tensor_from_ir_node(node, ir);
}

/** Elementwise a + b, broadcast and dtype-promoted (lazy). @see uop_add */
Tensor* tensor_add(Tensor* a, Tensor* b) { return uop_add(a, b); }

/** Elementwise a - b, broadcast and dtype-promoted (lazy). @see uop_sub */
Tensor* tensor_sub(Tensor* a, Tensor* b) { return uop_sub(a, b); }

/** Elementwise a * b, broadcast and dtype-promoted (lazy). @see uop_mul */
Tensor* tensor_mul(Tensor* a, Tensor* b) { return uop_mul(a, b); }

/** Elementwise a / b, broadcast and dtype-promoted (lazy). @see uop_div */
Tensor* tensor_div(Tensor* a, Tensor* b) { return uop_div(a, b); }

/** Elementwise a ^ b (lazy). @see uop_pow */
Tensor* tensor_pow(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_pow(a, b);
}

/** Elementwise negation (lazy). */
Tensor* tensor_neg(Tensor* a) { return forward_unary(a, UOP_NEG); }

/** Elementwise natural exponential (lazy). */
Tensor* tensor_exp(Tensor* a) { return forward_unary(a, UOP_EXP); }

/** Elementwise natural logarithm (lazy). */
Tensor* tensor_log(Tensor* a) { return forward_unary(a, UOP_LOG); }

/** Elementwise square root (lazy). */
Tensor* tensor_sqrt(Tensor* a) { return forward_unary(a, UOP_SQRT); }

/** Elementwise sine (lazy). @see uop_sin */
Tensor* tensor_sin(Tensor* a) {
    if (!a)
        return NULL;
    return uop_sin(a);
}

/** Elementwise cosine (lazy). @see uop_cos */
Tensor* tensor_cos(Tensor* a) {
    if (!a)
        return NULL;
    return uop_cos(a);
}

/** Elementwise tangent (lazy). @see uop_tan */
Tensor* tensor_tan(Tensor* a) {
    if (!a)
        return NULL;
    return uop_tan(a);
}

/** Elementwise hyperbolic tangent activation (lazy). @see uop_tanh */
Tensor* tensor_tanh(Tensor* a) {
    if (!a)
        return NULL;
    return uop_tanh(a);
}

/** ReLU activation, max(0, a) (lazy). @see uop_relu */
Tensor* tensor_relu(Tensor* a) {
    if (!a)
        return NULL;
    return uop_relu(a);
}

/** Logistic sigmoid activation (lazy). @see uop_sigmoid */
Tensor* tensor_sigmoid(Tensor* a) {
    if (!a)
        return NULL;
    return uop_sigmoid(a);
}

/** Leaky ReLU with the given negative-region slope (lazy). @see uop_leaky_relu */
Tensor* tensor_leaky_relu(Tensor* a, float negative_slope) {
    if (!a)
        return NULL;
    return uop_leaky_relu(a, negative_slope);
}

/** Softmax over dim (negative indices count from the end); errors on out-of-range dim. */
Tensor* tensor_softmax(Tensor* a, int dim) {
    if (!a)
        return NULL;

    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = a->ndim + normalized_dim;
    }
    if (normalized_dim < 0 || normalized_dim >= a->ndim) {
        LOG_ERROR("Softmax: dimension %d out of range for %dD tensor", dim, a->ndim);
        error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
        return NULL;
    }

    return uop_softmax(a, normalized_dim);
}

/* Resolve a reduction axis for the uop_*_dim family.
 *
 * -1 is this API's long-standing sentinel for "reduce every axis" -- NOT "the
 * last axis" as in PyTorch. Callers depend on it (zoo/clip.c reduces a loss to a
 * scalar with dim=-1, as does torch/pte.c), so it is preserved.
 *
 * What was wrong: the old guard `dim < a->ndim ? dim : -1` tested only the upper
 * bound, so every other negative axis fell through unchanged and hit the same
 * reduce-all sentinel downstream. `sum(x, -2)` on a [2,3] tensor silently
 * returned a scalar instead of reducing axis 0. Those now count from the end,
 * and a genuinely out-of-range axis still falls back to reduce-all rather than
 * indexing past the shape. */
static int resolve_reduce_dim(const Tensor* a, int dim) {
    if (dim == -1)
        return -1; /* documented reduce-all sentinel */
    if (dim < 0)
        dim += a->ndim;
    return (dim >= 0 && dim < a->ndim) ? dim : -1;
}

/** Sum over dim (dim=-1 reduces all axes); see resolve_reduce_dim for axis handling. */
Tensor* tensor_sum(Tensor* a, int dim, bool keepdim) {
    if (!a)
        return NULL;
    return uop_sum_dim(a, resolve_reduce_dim(a, dim), keepdim);
}

/** Mean over dim (dim=-1 reduces all axes); see resolve_reduce_dim for axis handling. */
Tensor* tensor_mean(Tensor* a, int dim, bool keepdim) {
    if (!a)
        return NULL;
    return uop_mean_dim(a, resolve_reduce_dim(a, dim), keepdim);
}

/** Max-reduce over dim (dim=-1 reduces all axes); see resolve_reduce_dim for axis handling. */
Tensor* tensor_max(Tensor* a, int dim, bool keepdim) {
    if (!a)
        return NULL;
    return uop_max_reduce_dim(a, resolve_reduce_dim(a, dim), keepdim);
}

/** Min over dim, computed as -max(-a) so it reuses the differentiable max-reduce path. */
Tensor* tensor_min(Tensor* a, int dim, bool keepdim) {
    if (!a)
        return NULL;

    /* min(a) = -max(-a) */
    Tensor* neg_a = uop_neg(a);
    if (!neg_a)
        return NULL;

    ReduceParams params;
    int* dims = NULL;
    if (dim >= 0 && dim < a->ndim) {
        dims = cml_malloc(sizeof(int));
        if (!dims) {
            tensor_free(neg_a);
            return NULL;
        }
        dims[0]         = dim;
        params.dims     = dims;
        params.num_dims = 1;
    } else {
        params.dims     = NULL;
        params.num_dims = 0;
    }
    params.keepdim = keepdim;

    Tensor* neg_max = uop_max_reduce(neg_a, &params);
    tensor_free(neg_a);

    if (!neg_max) {
        if (dims)
            cml_free(dims);
        return NULL;
    }

    Tensor* result = uop_neg(neg_max);
    tensor_free(neg_max);

    if (dims)
        cml_free(dims);
    return result;
}

/** Swap axes dim0 and dim1 via a permutation (lazy); negative dims default to the last two
 * axes. Errors on out-of-range dims. */
Tensor* tensor_transpose(Tensor* a, int dim0, int dim1) {
    if (!a)
        return NULL;

    if (dim0 < 0)
        dim0 = a->ndim >= 2 ? a->ndim - 2 : 0;
    if (dim1 < 0)
        dim1 = a->ndim >= 2 ? a->ndim - 1 : 0;

    if (dim0 >= a->ndim || dim1 >= a->ndim || dim0 < 0 || dim1 < 0) {
        LOG_ERROR("Transpose: invalid dimensions %d, %d for tensor with ndim=%d", dim0, dim1,
                  a->ndim);
        error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
        return NULL;
    }

    int* perm = cml_malloc((size_t)a->ndim * sizeof(int));
    if (!perm)
        return NULL;

    for (int i = 0; i < a->ndim; i++) {
        perm[i] = i;
    }
    perm[dim0] = dim1;
    perm[dim1] = dim0;

    PermuteParams params;
    params.perm     = perm;
    params.num_dims = a->ndim;

    Tensor* result = uop_permute(a, &params);

    cml_free(perm);
    return result;
}

/** Batched matrix multiply recorded into the lazy graph; batch dims come from a, the matrix
 * dims from a's rows and b's columns. Requires both operands be at least 2D. */
Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    if (a->ndim < 2 || b->ndim < 2) {
        CML_ERR_NULL("MatMul requires at least 2D tensors");
    }
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_MATMUL, inputs, 2, NULL) != 0)
        return NULL;
    struct IRNode* node = cml_ir_get_tail(ir);
    int batch_dims      = (a->ndim > 2) ? a->ndim - 2 : 0;
    int out_ndim        = batch_dims + 2;
    node->output_shape  = cml_malloc((size_t)out_ndim * sizeof(int));
    if (!node->output_shape) {
        CML_ERR_NULL("Failed to allocate output shape for matmul");
    }
    for (int i = 0; i < batch_dims; i++) {
        node->output_shape[i] = a->shape[i];
    }
    node->output_shape[out_ndim - 2] = a->shape[a->ndim - 2];
    node->output_shape[out_ndim - 1] = b->shape[b->ndim - 1];
    node->output_ndim                = out_ndim;
    if (a->requires_grad || b->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = a->requires_grad;
        node->needs_input_grad[1] = b->requires_grad;
    }
    return tensor_from_ir_node(node, ir);
}

/** Variance over dim as mean of squared deviations; unbiased divides by N-1 (guarded against
 * N<=1), biased by N. Built from sub/mul/reduce so it is differentiable. */
Tensor* tensor_var(Tensor* a, int dim, bool unbiased, bool keepdim) {
    if (!a)
        return NULL;

    Tensor* mean_val = tensor_mean(a, dim, true);
    if (!mean_val)
        return NULL;

    Tensor* diff = uop_sub(a, mean_val);
    if (!diff) {
        tensor_free(mean_val);
        return NULL;
    }

    Tensor* sq_diff = uop_mul(diff, diff);
    if (!sq_diff) {
        tensor_free(diff);
        tensor_free(mean_val);
        return NULL;
    }

    Tensor* result = NULL;

    if (!unbiased) {
        result = tensor_mean(sq_diff, dim, keepdim);
    } else {
        Tensor* sum_sq = tensor_sum(sq_diff, dim, keepdim);
        if (!sum_sq) {
            tensor_free(sq_diff);
            tensor_free(diff);
            tensor_free(mean_val);
            return NULL;
        }

        size_t N;
        if (dim < 0) {
            N = a->numel;
        } else {
            N = (size_t)a->shape[dim];
        }

        float n_minus_1 = (float)(N - 1);
        if (n_minus_1 <= 0.0f)
            n_minus_1 = 1.0f; // Guard against division by zero

        int scalar_shape[] = {1};
        Tensor* divisor    = tensor_full(scalar_shape, 1, NULL, n_minus_1);
        if (!divisor) {
            tensor_free(sum_sq);
            tensor_free(sq_diff);
            tensor_free(diff);
            tensor_free(mean_val);
            return NULL;
        }

        result = uop_div(sum_sq, divisor);

        tensor_free(divisor);
        tensor_free(sum_sq);
    }

    tensor_free(sq_diff);
    tensor_free(diff);
    tensor_free(mean_val);
    return result;
}

/** Standard deviation over dim, as sqrt of tensor_var. */
Tensor* tensor_std(Tensor* a, int dim, bool unbiased, bool keepdim) {
    if (!a)
        return NULL;

    Tensor* var_tensor = tensor_var(a, dim, unbiased, keepdim);
    if (!var_tensor)
        return NULL;

    Tensor* result = uop_sqrt(var_tensor);
    tensor_free(var_tensor);
    return result;
}

/** Indices of the maxima over dim (dim<0 reduces all axes). Non-differentiable. */
Tensor* tensor_argmax(Tensor* a, int dim) {
    if (!a)
        return NULL;
    ReduceParams params = {0};
    if (dim >= 0) {
        params.dims     = &dim;
        params.num_dims = 1;
    }
    return uop_argmax(a, dim >= 0 ? &params : NULL);
}

/** Indices of the minima over dim (dim<0 reduces all axes). Non-differentiable. */
Tensor* tensor_argmin(Tensor* a, int dim) {
    if (!a)
        return NULL;
    ReduceParams params = {0};
    if (dim >= 0) {
        params.dims     = &dim;
        params.num_dims = 1;
    }
    return uop_argmin(a, dim >= 0 ? &params : NULL);
}

/** Whether the tensor currently holds an accumulated gradient. */
bool tensor_has_grad(Tensor* a) { return a && a->grad != NULL; }

/** ELU activation with the given alpha (lazy). @see uop_elu */
Tensor* tensor_elu(Tensor* a, float alpha) {
    if (!a)
        return NULL;
    return uop_elu(a, alpha);
}

/** SELU (scaled ELU) activation (lazy). @see uop_selu */
Tensor* tensor_selu(Tensor* a) {
    if (!a)
        return NULL;
    return uop_selu(a);
}

/** Mish activation (lazy). @see uop_mish */
Tensor* tensor_mish(Tensor* a) {
    if (!a)
        return NULL;
    return uop_mish(a);
}

/** SiLU / swish activation (lazy). @see uop_silu */
Tensor* tensor_silu(Tensor* a) {
    if (!a)
        return NULL;
    return uop_silu(a);
}

/** Hard-swish activation (lazy). @see uop_hardswish */
Tensor* tensor_hardswish(Tensor* a) {
    if (!a)
        return NULL;
    return uop_hardswish(a);
}

/** Sort values along dim, ascending or descending (lazy). @see uop_sort */
Tensor* tensor_sort(Tensor* a, int dim, bool descending) {
    if (!a)
        return NULL;
    return uop_sort(a, dim, descending);
}

/** Top-k values along dim; the sorted flag is ignored since results are always sorted. */
Tensor* tensor_topk(Tensor* a, int k, int dim, bool largest, bool sorted) {
    (void)sorted; // topk always returns sorted
    if (!a)
        return NULL;
    return uop_topk(a, k, dim, largest, NULL);
}

/** Select the elements of a where mask is true, flattened (lazy). @see uop_masked_select */
Tensor* tensor_masked_select(Tensor* a, Tensor* mask) {
    if (!a || !mask)
        return NULL;
    return uop_masked_select(a, mask);
}

/** Build coordinate grids from the input 1-D tensors; writes the output count to
 * num_outputs and returns a newly allocated array of tensors. @see uop_meshgrid */
Tensor** tensor_meshgrid(Tensor** tensors, int num_tensors, int* num_outputs) {
    if (!tensors || !num_outputs)
        return NULL;
    return uop_meshgrid(tensors, num_tensors, num_outputs);
}

/** Extract the diagonal spanning dim1 and dim2 at the given offset (lazy). @see uop_diagonal */
Tensor* tensor_diagonal(Tensor* a, int offset, int dim1, int dim2) {
    if (!a)
        return NULL;
    return uop_diagonal(a, offset, dim1, dim2);
}

/** Linear interpolation a + weight*(b - a), with weight broadcast as a filled tensor (lazy). */
Tensor* tensor_lerp(Tensor* a, Tensor* b, float weight) {
    if (!a || !b)
        return NULL;
    Tensor* w = uop_fill(a->shape, a->ndim, weight);
    if (!w)
        return NULL;
    return uop_lerp(a, b, w);
}

/** Elementwise integer (floor) division (lazy). @see uop_idiv */
Tensor* tensor_idiv(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_idiv(a, b);
}

/** Elementwise modulo (lazy). @see uop_mod */
Tensor* tensor_mod(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_mod(a, b);
}
