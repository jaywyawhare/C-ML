#include "autograd/autograd.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to tensor_add");
        return NULL;
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        LOG_ERROR("Failed to get or create IR context");
        return NULL;
    }

    Tensor* inputs[]    = {a, b};
    struct IRNode* node = NULL;
    if (cml_ir_add_uop(ir, UOP_ADD, inputs, 2, NULL) != 0)
        return NULL;
    node = cml_ir_get_tail(ir);

    if (cml_ir_compute_broadcast_shape(node) != 0)
        return NULL;

    if (a->requires_grad || b->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = a->requires_grad;
        node->needs_input_grad[1] = b->requires_grad;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to tensor_sub");
        return NULL;
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_SUB, inputs, 2, NULL) != 0)
        return NULL;
    struct IRNode* node = cml_ir_get_tail(ir);

    if (cml_ir_compute_broadcast_shape(node) != 0)
        return NULL;

    if (a->requires_grad || b->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = a->requires_grad;
        node->needs_input_grad[1] = b->requires_grad;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_MUL, inputs, 2, NULL) != 0)
        return NULL;
    struct IRNode* node = cml_ir_get_tail(ir);
    if (cml_ir_compute_broadcast_shape(node) != 0)
        return NULL;
    if (a->requires_grad || b->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = a->requires_grad;
        node->needs_input_grad[1] = b->requires_grad;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* tensor_div(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_DIV, inputs, 2, NULL) != 0)
        return NULL;
    struct IRNode* node = cml_ir_get_tail(ir);
    if (cml_ir_compute_broadcast_shape(node) != 0)
        return NULL;
    if (a->requires_grad || b->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = a->requires_grad;
        node->needs_input_grad[1] = b->requires_grad;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* tensor_pow(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_pow(a, b);
}

Tensor* tensor_neg(Tensor* a) {
    if (!a)
        return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_NEG, inputs, 1, NULL) != 0)
        return NULL;
    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim  = a->ndim;
    if (a->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* tensor_exp(Tensor* a) {
    if (!a)
        return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_EXP, inputs, 1, NULL) != 0)
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

Tensor* tensor_log(Tensor* a) {
    if (!a)
        return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_LOG, inputs, 1, NULL) != 0)
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

Tensor* tensor_sqrt(Tensor* a) {
    if (!a)
        return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_SQRT, inputs, 1, NULL) != 0)
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

Tensor* tensor_sin(Tensor* a) {
    if (!a)
        return NULL;
    return uop_sin(a);
}

Tensor* tensor_cos(Tensor* a) {
    if (!a)
        return NULL;
    return uop_cos(a);
}

Tensor* tensor_tan(Tensor* a) {
    if (!a)
        return NULL;
    return uop_tan(a);
}

Tensor* tensor_tanh(Tensor* a) {
    if (!a)
        return NULL;
    return uop_tanh(a);
}

Tensor* tensor_relu(Tensor* a) {
    if (!a)
        return NULL;
    return uop_relu(a);
}

Tensor* tensor_sigmoid(Tensor* a) {
    if (!a)
        return NULL;
    return uop_sigmoid(a);
}

Tensor* tensor_leaky_relu(Tensor* a, float negative_slope) {
    if (!a)
        return NULL;
    return uop_leaky_relu(a, negative_slope);
}

Tensor* tensor_softmax(Tensor* a, int dim) {
    if (!a)
        return NULL;

    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = a->ndim + normalized_dim;
    }
    if (normalized_dim < 0 || normalized_dim >= a->ndim) {
        LOG_ERROR("Softmax: dimension %d out of range for %dD tensor", dim, a->ndim);
        return NULL;
    }

    return uop_softmax(a, normalized_dim);
}

Tensor* tensor_sum(Tensor* a, int dim, bool keepdim) {
    if (!a)
        return NULL;

    ReduceParams params;
    int* dims = NULL;
    if (dim >= 0 && dim < a->ndim) {
        dims = malloc(sizeof(int));
        if (!dims)
            return NULL;
        dims[0]         = dim;
        params.dims     = dims;
        params.num_dims = 1;
    } else {
        params.dims     = NULL;
        params.num_dims = 0;
    }
    params.keepdim = keepdim;

    Tensor* result = uop_sum(a, &params);

    if (dims)
        free(dims);
    return result;
}

Tensor* tensor_mean(Tensor* a, int dim, bool keepdim) {
    if (!a)
        return NULL;

    ReduceParams params;
    int* dims = NULL;
    if (dim >= 0 && dim < a->ndim) {
        dims = malloc(sizeof(int));
        if (!dims)
            return NULL;
        dims[0]         = dim;
        params.dims     = dims;
        params.num_dims = 1;
    } else {
        params.dims     = NULL;
        params.num_dims = 0;
    }
    params.keepdim = keepdim;

    Tensor* result = uop_mean(a, &params);

    if (dims)
        free(dims);
    return result;
}

Tensor* tensor_max(Tensor* a, int dim, bool keepdim) {
    if (!a)
        return NULL;

    ReduceParams params;
    int* dims = NULL;
    if (dim >= 0 && dim < a->ndim) {
        dims = malloc(sizeof(int));
        if (!dims)
            return NULL;
        dims[0]         = dim;
        params.dims     = dims;
        params.num_dims = 1;
    } else {
        params.dims     = NULL;
        params.num_dims = 0;
    }
    params.keepdim = keepdim;

    Tensor* result = uop_max_reduce(a, &params);

    if (dims)
        free(dims);
    return result;
}

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
        dims = malloc(sizeof(int));
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
            free(dims);
        return NULL;
    }

    Tensor* result = uop_neg(neg_max);
    tensor_free(neg_max);

    if (dims)
        free(dims);
    return result;
}

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
        return NULL;
    }

    int* perm = malloc((size_t)a->ndim * sizeof(int));
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

    free(perm);
    return result;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    if (a->ndim < 2 || b->ndim < 2) {
        LOG_ERROR("MatMul requires at least 2D tensors");
        return NULL;
    }
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_MATMUL, inputs, 2, NULL) != 0)
        return NULL;
    struct IRNode* node = cml_ir_get_tail(ir);
    int M                 = a->shape[a->ndim - 2];
    int N                 = b->shape[b->ndim - 1];
    node->output_shape    = malloc(2 * sizeof(int));
    node->output_shape[0] = M;
    node->output_shape[1] = N;
    node->output_ndim     = 2;
    if (a->requires_grad || b->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = a->requires_grad;
        node->needs_input_grad[1] = b->requires_grad;
    }
    return tensor_from_ir_node(node, ir);
}

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

Tensor* tensor_argmax(Tensor* a, int dim) {
    if (!a)
        return NULL;

    tensor_ensure_executed(a);
    float* data = (float*)tensor_data_ptr(a);
    if (!data)
        return NULL;

    TensorConfig config = {
        .dtype = DTYPE_INT32, .has_dtype = true, .device = a->device, .has_device = true};

    if (dim < 0) {
        float max_val = -FLT_MAX;
        int max_idx   = 0;
        for (size_t i = 0; i < a->numel; i++) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = (int)i;
            }
        }

        int shape[]    = {1};
        Tensor* result = tensor_empty(shape, 1, &config);
        if (!result)
            return NULL;
        tensor_ensure_executed(result);
        int* result_data = (int*)tensor_data_ptr(result);
        result_data[0]   = max_idx;
        return result;
    }
    if (dim >= a->ndim) {
        LOG_ERROR("tensor_argmax: dimension %d out of range for %dD tensor", dim, a->ndim);
        return NULL;
    }
    int out_ndim = a->ndim - 1;
    if (out_ndim == 0)
        out_ndim = 1;

    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape)
        return NULL;

    if (a->ndim == 1) {
        out_shape[0] = 1;
    } else {
        int j = 0;
        for (int i = 0; i < a->ndim; i++) {
            if (i != dim)
                out_shape[j++] = a->shape[i];
        }
    }

    Tensor* result = tensor_empty(out_shape, out_ndim, &config);
    if (!result) {
        free(out_shape);
        return NULL;
    }
    tensor_ensure_executed(result);
    int* result_data = (int*)tensor_data_ptr(result);
    size_t outer_size = 1;
    for (int i = 0; i < dim; i++)
        outer_size *= (size_t)a->shape[i];

    size_t dim_size = (size_t)a->shape[dim];

    size_t inner_size = 1;
    for (int i = dim + 1; i < a->ndim; i++)
        inner_size *= (size_t)a->shape[i];
    for (size_t o = 0; o < outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            float max_val = -FLT_MAX;
            int max_idx   = 0;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + in;
                if (data[idx] > max_val) {
                    max_val = data[idx];
                    max_idx = (int)d;
                }
            }
            result_data[o * inner_size + in] = max_idx;
        }
    }

    free(out_shape);
    return result;
}

Tensor* tensor_argmin(Tensor* a, int dim) {
    if (!a)
        return NULL;

    // Eager execution: we need actual data
    tensor_ensure_executed(a);
    float* data = (float*)tensor_data_ptr(a);
    if (!data)
        return NULL;

    TensorConfig config = {
        .dtype = DTYPE_INT32, .has_dtype = true, .device = a->device, .has_device = true};

    if (dim < 0) {
        float min_val = FLT_MAX;
        int min_idx   = 0;
        for (size_t i = 0; i < a->numel; i++) {
            if (data[i] < min_val) {
                min_val = data[i];
                min_idx = (int)i;
            }
        }

        int shape[]    = {1};
        Tensor* result = tensor_empty(shape, 1, &config);
        if (!result)
            return NULL;
        tensor_ensure_executed(result);
        int* result_data = (int*)tensor_data_ptr(result);
        result_data[0]   = min_idx;
        return result;
    }
    if (dim >= a->ndim) {
        LOG_ERROR("tensor_argmin: dimension %d out of range for %dD tensor", dim, a->ndim);
        return NULL;
    }
    int out_ndim = a->ndim - 1;
    if (out_ndim == 0)
        out_ndim = 1;

    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape)
        return NULL;

    if (a->ndim == 1) {
        out_shape[0] = 1;
    } else {
        int j = 0;
        for (int i = 0; i < a->ndim; i++) {
            if (i != dim)
                out_shape[j++] = a->shape[i];
        }
    }

    Tensor* result = tensor_empty(out_shape, out_ndim, &config);
    if (!result) {
        free(out_shape);
        return NULL;
    }
    tensor_ensure_executed(result);
    int* result_data = (int*)tensor_data_ptr(result);
    size_t outer_size = 1;
    for (int i = 0; i < dim; i++)
        outer_size *= (size_t)a->shape[i];

    size_t dim_size = (size_t)a->shape[dim];

    size_t inner_size = 1;
    for (int i = dim + 1; i < a->ndim; i++)
        inner_size *= (size_t)a->shape[i];
    for (size_t o = 0; o < outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            float min_val = FLT_MAX;
            int min_idx   = 0;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + in;
                if (data[idx] < min_val) {
                    min_val = data[idx];
                    min_idx = (int)d;
                }
            }
            result_data[o * inner_size + in] = min_idx;
        }
    }

    free(out_shape);
    return result;
}

bool tensor_has_grad(Tensor* a) { return a && a->grad != NULL; }

Tensor* tensor_elu(Tensor* a, float alpha) {
    if (!a) return NULL;
    return uop_elu(a, alpha);
}

Tensor* tensor_selu(Tensor* a) {
    if (!a) return NULL;
    return uop_selu(a);
}

Tensor* tensor_mish(Tensor* a) {
    if (!a) return NULL;
    return uop_mish(a);
}

Tensor* tensor_silu(Tensor* a) {
    if (!a) return NULL;
    return uop_silu(a);
}

Tensor* tensor_hardswish(Tensor* a) {
    if (!a) return NULL;
    return uop_hardswish(a);
}

Tensor* tensor_sort(Tensor* a, int dim, bool descending) {
    if (!a) return NULL;
    return uop_sort(a, dim, descending);
}

Tensor* tensor_topk(Tensor* a, int k, int dim, bool largest, bool sorted) {
    (void)sorted; // topk always returns sorted
    if (!a) return NULL;
    return uop_topk(a, k, dim, largest, NULL);
}

Tensor* tensor_masked_select(Tensor* a, Tensor* mask) {
    if (!a || !mask) return NULL;
    return uop_masked_select(a, mask);
}

Tensor** tensor_meshgrid(Tensor** tensors, int num_tensors, int* num_outputs) {
    if (!tensors || !num_outputs) return NULL;
    return uop_meshgrid(tensors, num_tensors, num_outputs);
}

Tensor* tensor_diagonal(Tensor* a, int offset, int dim1, int dim2) {
    if (!a) return NULL;
    return uop_diagonal(a, offset, dim1, dim2);
}

Tensor* tensor_lerp(Tensor* a, Tensor* b, float weight) {
    if (!a || !b) return NULL;
    Tensor* w = uop_fill(a->shape, a->ndim, weight);
    if (!w) return NULL;
    return uop_lerp(a, b, w);
}

Tensor* tensor_idiv(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    return uop_idiv(a, b);
}

Tensor* tensor_mod(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    return uop_mod(a, b);
}
