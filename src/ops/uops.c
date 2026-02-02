/**
 * @file uops.c
 * @brief Micro-Operations implementation
 */

#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include "core/error_stack.h"
#include "core/error_codes.h"
#include "tensor/tensor.h"
#include "autograd/autograd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Forward declaration

Tensor* uop_add(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    // UOps ONLY create IR nodes, never execute!
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_ADD, inputs, 2, NULL) != 0)
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

Tensor* uop_sub(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
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

Tensor* uop_mul(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
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

Tensor* uop_div(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
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

Tensor* uop_max(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to uop_max");
        return NULL;
    }

    // Check if shapes can be broadcast
    if (!tensor_can_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim)) {
        LOG_ERROR("Shapes cannot be broadcast for uop_max");
        return NULL;
    }

    // Compute output shape
    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape)
        return NULL;

    // Lazy: use IR
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        free(out_shape);
        return NULL;
    }

    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_MAX, inputs, 2, NULL) != 0) {
        free(out_shape);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape  = out_shape;
    node->output_ndim   = out_ndim;

    if (a->requires_grad || b->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = a->requires_grad;
        node->needs_input_grad[1] = b->requires_grad;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_cmplt(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to uop_cmplt");
        return NULL;
    }

    // Check if shapes can be broadcast
    if (!tensor_can_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim)) {
        LOG_ERROR("Shapes cannot be broadcast for uop_cmplt");
        return NULL;
    }

    // Compute output shape
    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape)
        return NULL;

    // UOps ONLY create IR nodes, never execute! (Lazy evaluation like PyTorch)
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        free(out_shape);
        return NULL;
    }

    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_CMPLT, inputs, 2, NULL) != 0) {
        free(out_shape);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape  = out_shape;
    node->output_ndim   = out_ndim;

    // CMPLT doesn't require grad (comparison operations don't have gradients)
    // but we still track if inputs require grad for the graph
    if (a->requires_grad || b->requires_grad) {
        node->requires_grad       = false; // Comparison result doesn't have gradient
        node->needs_input_grad[0] = false;
        node->needs_input_grad[1] = false;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_neg(Tensor* a) {
    if (!a)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_NEG, inputs, 1, NULL) != 0)
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

Tensor* uop_exp(Tensor* a) {
    if (!a)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
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

Tensor* uop_log(Tensor* a) {
    if (!a)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
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

Tensor* uop_sqrt(Tensor* a) {
    if (!a)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
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

Tensor* uop_recip(Tensor* a) {
    if (!a) {
        LOG_ERROR("NULL tensor input to uop_recip");
        return NULL;
    }

    // Create ones tensor lazily for division
    Tensor* ones = uop_fill(a->shape, a->ndim, 1.0f);
    if (!ones)
        return NULL;

    // 1 / a = ones / a (lazy)
    Tensor* result = uop_div(ones, a);
    // Don't free ones - it's part of the IR graph
    return result;
}

Tensor* uop_abs(Tensor* a) {
    if (!a)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_ABS, inputs, 1, NULL) != 0)
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

Tensor* uop_sin(Tensor* a) {
    if (!a)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_SIN, inputs, 1, NULL) != 0)
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

Tensor* uop_cos(Tensor* a) {
    if (!a)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_COS, inputs, 1, NULL) != 0)
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

Tensor* uop_tan(Tensor* a) {
    if (!a)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_TAN, inputs, 1, NULL) != 0)
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

Tensor* uop_pow(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_POW, inputs, 2, NULL) != 0)
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

Tensor* uop_sum(Tensor* a, ReduceParams* params) {
    if (!a) {
        LOG_ERROR("NULL tensor input to uop_sum");
        return NULL;
    }

    // UOps ONLY create IR nodes, never execute!
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_SUM, inputs, 1, params) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);

    // Compute output shape from reduction params
    int dim      = params && params->dims && params->num_dims > 0 ? params->dims[0] : -1;
    bool keepdim = params ? params->keepdim : false;

    int* out_shape = NULL;
    int out_ndim   = 0;

    if (dim < 0 || dim >= a->ndim) {
        // Reduce all dimensions
        out_ndim  = 1;
        out_shape = malloc(sizeof(int));
        if (!out_shape)
            return NULL;
        out_shape[0] = 1;
    } else {
        // Reduce along specified dimension
        out_ndim = keepdim ? a->ndim : (a->ndim - 1);
        if (out_ndim == 0)
            out_ndim = 1;
        out_shape = malloc((size_t)out_ndim * sizeof(int));
        if (!out_shape)
            return NULL;

        int out_idx = 0;
        if (keepdim) {
            for (int i = 0; i < a->ndim; i++) {
                out_shape[i] = (i == dim) ? 1 : a->shape[i];
            }
        } else {
            for (int i = 0; i < a->ndim; i++) {
                if (i != dim) {
                    out_shape[out_idx++] = a->shape[i];
                }
            }
        }
    }

    node->output_shape = out_shape;
    node->output_ndim  = out_ndim;

    // Set autograd
    if (a->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_max_reduce(Tensor* a, ReduceParams* params) {
    if (!a) {
        LOG_ERROR("NULL tensor input to uop_max_reduce");
        return NULL;
    }

    // UOps ONLY create IR nodes, never execute! (Lazy evaluation like PyTorch)
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    // Deep copy params for IR node ownership
    ReduceParams* new_params = malloc(sizeof(ReduceParams));
    if (!new_params)
        return NULL;

    new_params->keepdim  = params ? params->keepdim : false;
    new_params->num_dims = params ? params->num_dims : 0;

    if (params && params->dims && params->num_dims > 0) {
        new_params->dims = malloc((size_t)params->num_dims * sizeof(int));
        if (!new_params->dims) {
            free(new_params);
            return NULL;
        }
        memcpy(new_params->dims, params->dims, (size_t)params->num_dims * sizeof(int));
    } else {
        new_params->dims     = NULL;
        new_params->num_dims = 0;
    }

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_MAX_REDUCE, inputs, 1, new_params) != 0) {
        if (new_params->dims)
            free(new_params->dims);
        free(new_params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);

    // Compute output shape from reduction params
    int dim      = params && params->dims && params->num_dims > 0 ? params->dims[0] : -1;
    bool keepdim = params ? params->keepdim : false;

    int* out_shape = NULL;
    int out_ndim   = 0;

    if (dim < 0 || dim >= a->ndim) {
        // Reduce all dimensions
        out_ndim  = 1;
        out_shape = malloc(sizeof(int));
        if (!out_shape)
            return NULL;
        out_shape[0] = 1;
    } else {
        // Reduce along specified dimension
        out_ndim = keepdim ? a->ndim : (a->ndim - 1);
        if (out_ndim == 0)
            out_ndim = 1;
        out_shape = malloc((size_t)out_ndim * sizeof(int));
        if (!out_shape)
            return NULL;

        int out_idx = 0;
        if (keepdim) {
            for (int i = 0; i < a->ndim; i++) {
                out_shape[i] = (i == dim) ? 1 : a->shape[i];
            }
        } else {
            for (int i = 0; i < a->ndim; i++) {
                if (i != dim) {
                    out_shape[out_idx++] = a->shape[i];
                }
            }
        }
    }

    node->output_shape = out_shape;
    node->output_ndim  = out_ndim;

    // Set autograd
    if (a->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_mean(Tensor* a, ReduceParams* params) {
    if (!a) {
        LOG_ERROR("NULL tensor input to uop_mean");
        return NULL;
    }

    // UOps ONLY create IR nodes, never execute!
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    // Deep copy params
    ReduceParams* new_params = malloc(sizeof(ReduceParams));
    if (!new_params)
        return NULL;

    new_params->keepdim  = params ? params->keepdim : false;
    new_params->num_dims = params ? params->num_dims : 0;

    if (params && params->dims && params->num_dims > 0) {
        new_params->dims = malloc((size_t)params->num_dims * sizeof(int));
        if (!new_params->dims) {
            free(new_params);
            return NULL;
        }
        memcpy(new_params->dims, params->dims, (size_t)params->num_dims * sizeof(int));
    } else {
        new_params->dims     = NULL;
        new_params->num_dims = 0;
    }

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_MEAN, inputs, 1, new_params) != 0) {
        if (new_params->dims)
            free(new_params->dims);
        free(new_params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);

    // Compute output shape from reduction params
    int dim      = params && params->dims && params->num_dims > 0 ? params->dims[0] : -1;
    bool keepdim = params ? params->keepdim : false;

    int* out_shape = NULL;
    int out_ndim   = 0;

    if (dim < 0 || dim >= a->ndim) {
        // Reduce all dimensions
        out_ndim  = 1;
        out_shape = malloc(sizeof(int));
        if (!out_shape)
            return NULL;
        out_shape[0] = 1;
    } else {
        // Reduce along specified dimension
        out_ndim = keepdim ? a->ndim : (a->ndim - 1);
        if (out_ndim == 0)
            out_ndim = 1;
        out_shape = malloc((size_t)out_ndim * sizeof(int));
        if (!out_shape)
            return NULL;

        int out_idx = 0;
        if (keepdim) {
            for (int i = 0; i < a->ndim; i++) {
                out_shape[i] = (i == dim) ? 1 : a->shape[i];
            }
        } else {
            for (int i = 0; i < a->ndim; i++) {
                if (i != dim) {
                    out_shape[out_idx++] = a->shape[i];
                }
            }
        }
    }

    node->output_shape = out_shape;
    node->output_ndim  = out_ndim;

    // Set autograd
    if (a->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_reshape(Tensor* a, ReshapeParams* params) {
    if (!a || !params) {
        LOG_ERROR("NULL input to uop_reshape");
        return NULL;
    }

    Tensor* result = tensor_reshape(a, params->new_shape, params->new_ndim);

    if (result) {
        CMLIR_t ir = cml_ir_get_or_create_context();
        if (ir) {
            // Deep copy params
            ReshapeParams* new_params = malloc(sizeof(ReshapeParams));
            if (new_params) {
                new_params->new_ndim  = params->new_ndim;
                new_params->new_shape = malloc((size_t)params->new_ndim * sizeof(int));
                if (new_params->new_shape) {
                    memcpy(new_params->new_shape, params->new_shape,
                           (size_t)params->new_ndim * sizeof(int));

                    Tensor* inputs[] = {a};
                    if (cml_ir_add_uop(ir, UOP_RESHAPE, inputs, 1, new_params) == 0) {
                        struct IRNode* node = cml_ir_get_tail(ir);
                        node->output_shape  = tensor_shape_copy(result->shape, result->ndim);
                        node->output_ndim   = result->ndim;
                        if (a->requires_grad) {
                            node->requires_grad       = true;
                            node->needs_input_grad[0] = true;
                        }
                        result->ir_node    = node;
                        result->ir_context = ir;
                        node->output       = result;
                    } else {
                        free(new_params->new_shape);
                        free(new_params);
                    }
                } else {
                    free(new_params);
                }
            }
        }
    }

    return result;
}

Tensor* uop_permute(Tensor* a, PermuteParams* params) {
    if (!a || !params || !params->perm) {
        LOG_ERROR("NULL input to uop_permute");
        return NULL;
    }

    if (params->num_dims != a->ndim) {
        LOG_ERROR("Permute: num_dims (%d) must match tensor ndim (%d)", params->num_dims, a->ndim);
        return NULL;
    }

    // Validate permutation
    bool* used = calloc((size_t)a->ndim, sizeof(bool));
    if (!used)
        return NULL;

    for (int i = 0; i < params->num_dims; i++) {
        if (params->perm[i] < 0 || params->perm[i] >= a->ndim) {
            free(used);
            LOG_ERROR("Permute: invalid dimension %d in permutation", params->perm[i]);
            return NULL;
        }
        if (used[params->perm[i]]) {
            free(used);
            LOG_ERROR("Permute: duplicate dimension %d in permutation", params->perm[i]);
            return NULL;
        }
        used[params->perm[i]] = true;
    }
    free(used);

    // UOps ONLY create IR nodes, never execute!
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    // Deep copy params for IR node
    PermuteParams* new_params = malloc(sizeof(PermuteParams));
    if (!new_params)
        return NULL;

    new_params->num_dims = params->num_dims;
    new_params->perm     = malloc((size_t)params->num_dims * sizeof(int));
    if (!new_params->perm) {
        free(new_params);
        return NULL;
    }
    memcpy(new_params->perm, params->perm, (size_t)params->num_dims * sizeof(int));

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_PERMUTE, inputs, 1, new_params) != 0) {
        free(new_params->perm);
        free(new_params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);

    // Compute output shape (permuted)
    int* out_shape = malloc((size_t)a->ndim * sizeof(int));
    if (!out_shape)
        return NULL;

    for (int i = 0; i < a->ndim; i++) {
        out_shape[i] = a->shape[params->perm[i]];
    }

    // Set output shape in IR node
    node->output_shape = out_shape;
    node->output_ndim  = a->ndim;

    // Set autograd
    if (a->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_expand(Tensor* a, ExpandParams* params) {
    if (!a || !params || !params->new_shape) {
        LOG_ERROR("NULL input to uop_expand");
        return NULL;
    }

    // Expand broadcasts tensor to new shape
    // Check if expansion is valid (can broadcast)
    if (params->new_ndim < a->ndim) {
        LOG_ERROR("Expand: new_ndim (%d) must be >= tensor ndim (%d)", params->new_ndim, a->ndim);
        return NULL;
    }

    // Build broadcast shape
    int* broadcast_shape = malloc((size_t)params->new_ndim * sizeof(int));
    if (!broadcast_shape)
        return NULL;

    // Prepend 1s for new dimensions
    int prepend = params->new_ndim - a->ndim;
    for (int i = 0; i < prepend; i++) {
        broadcast_shape[i] = 1;
    }

    // Copy original shape
    for (int i = 0; i < a->ndim; i++) {
        broadcast_shape[prepend + i] = a->shape[i];
    }

    // Validate broadcast compatibility
    for (int i = 0; i < params->new_ndim; i++) {
        int orig_dim = i >= prepend ? a->shape[i - prepend] : 1;
        if (orig_dim != params->new_shape[i] && orig_dim != 1 && params->new_shape[i] != 1) {
            free(broadcast_shape);
            LOG_ERROR("Expand: cannot broadcast dimension %d: %d -> %d", i, orig_dim,
                      params->new_shape[i]);
            return NULL;
        }
    }

    // Create view with expanded shape using tensor_as_strided
    // For broadcasting, we use stride 0 for dimensions that are 1
    size_t* new_strides = malloc((size_t)params->new_ndim * sizeof(size_t));
    if (!new_strides) {
        free(broadcast_shape);
        return NULL;
    }

    // Compute strides for broadcast
    size_t* orig_strides = a->strides ? a->strides : compute_contiguous_strides(a->shape, a->ndim);
    if (!orig_strides) {
        free(broadcast_shape);
        free(new_strides);
        return NULL;
    }

    new_strides[params->new_ndim - 1] = 1;
    for (int i = params->new_ndim - 2; i >= 0; i--) {
        if (i < prepend) {
            // New dimension: stride 0 (broadcast)
            new_strides[i] = 0;
        } else {
            int orig_idx = i - prepend;
            if (broadcast_shape[i] == 1 && params->new_shape[i] != 1) {
                // Broadcasting: stride 0
                new_strides[i] = 0;
            } else {
                // Use original stride
                new_strides[i] = orig_strides[orig_idx] *
                                 ((size_t)params->new_shape[i] / (size_t)broadcast_shape[i]);
            }
        }
    }

    if (!a->strides) {
        free(orig_strides);
    }

    // Create view using tensor_as_strided
    Tensor* result =
        tensor_as_strided(a, params->new_shape, params->new_ndim, new_strides, a->storage_offset);

    free(broadcast_shape);
    free(new_strides);

    if (result) {
        CMLIR_t ir = cml_ir_get_or_create_context();
        if (ir) {
            // Deep copy params
            ExpandParams* new_params = malloc(sizeof(ExpandParams));
            if (new_params) {
                new_params->new_ndim  = params->new_ndim;
                new_params->new_shape = malloc((size_t)params->new_ndim * sizeof(int));
                if (new_params->new_shape) {
                    memcpy(new_params->new_shape, params->new_shape,
                           (size_t)params->new_ndim * sizeof(int));

                    Tensor* inputs[] = {a};
                    if (cml_ir_add_uop(ir, UOP_EXPAND, inputs, 1, new_params) == 0) {
                        struct IRNode* node = cml_ir_get_tail(ir);
                        node->output_shape  = tensor_shape_copy(result->shape, result->ndim);
                        node->output_ndim   = result->ndim;
                        if (a->requires_grad) {
                            node->requires_grad       = true;
                            node->needs_input_grad[0] = true;
                        }
                        result->ir_node    = node;
                        result->ir_context = ir;
                        node->output       = result;
                    } else {
                        free(new_params->new_shape);
                        free(new_params);
                    }
                } else {
                    free(new_params);
                }
            }
        }
    }

    return result;
}

Tensor* uop_stride(Tensor* a, StrideParams* params) {
    if (!a || !params || !params->new_strides) {
        LOG_ERROR("NULL input to uop_stride");
        return NULL;
    }

    if (params->num_dims != a->ndim) {
        LOG_ERROR("Stride: num_dims (%d) must match tensor ndim (%d)", params->num_dims, a->ndim);
        return NULL;
    }

    // Create view with new strides using tensor_as_strided
    Tensor* result =
        tensor_as_strided(a, a->shape, a->ndim, params->new_strides, a->storage_offset);

    if (result) {
        CMLIR_t ir = cml_ir_get_or_create_context();
        if (ir) {
            // Deep copy params
            StrideParams* new_params = malloc(sizeof(StrideParams));
            if (new_params) {
                new_params->num_dims    = params->num_dims;
                new_params->new_strides = malloc((size_t)params->num_dims * sizeof(size_t));
                if (new_params->new_strides) {
                    memcpy(new_params->new_strides, params->new_strides,
                           (size_t)params->num_dims * sizeof(size_t));

                    Tensor* inputs[] = {a};
                    if (cml_ir_add_uop(ir, UOP_STRIDE, inputs, 1, new_params) == 0) {
                        struct IRNode* node = cml_ir_get_tail(ir);
                        node->output_shape  = tensor_shape_copy(result->shape, result->ndim);
                        node->output_ndim   = result->ndim;
                        if (a->requires_grad) {
                            node->requires_grad       = true;
                            node->needs_input_grad[0] = true;
                        }
                        result->ir_node    = node;
                        result->ir_context = ir;
                        node->output       = result;
                    } else {
                        free(new_params->new_strides);
                        free(new_params);
                    }
                } else {
                    free(new_params);
                }
            }
        }
    }

    return result;
}

Tensor* uop_slice(Tensor* a, SliceParams* params) {
    if (!a || !params || !params->start || !params->end) {
        LOG_ERROR("NULL input to uop_slice");
        return NULL;
    }

    if (params->num_dims != a->ndim) {
        LOG_ERROR("Slice: num_dims (%d) must match tensor ndim (%d)", params->num_dims, a->ndim);
        return NULL;
    }

    // Calculate new shape and offset
    int* new_shape = malloc((size_t)a->ndim * sizeof(int));
    if (!new_shape)
        return NULL;

    size_t storage_offset = a->storage_offset;

    size_t* strides   = a->strides ? a->strides : compute_contiguous_strides(a->shape, a->ndim);
    bool free_strides = !a->strides;

    if (!strides) {
        free(new_shape);
        return NULL;
    }

    for (int i = 0; i < a->ndim; i++) {
        int start = params->start[i];
        int end   = params->end[i];
        int step  = params->step ? params->step[i] : 1;

        // Adjust start/end for negative indices
        if (start < 0)
            start += a->shape[i];
        if (end < 0)
            end += a->shape[i];

        // Clamp
        if (start < 0)
            start = 0;
        if (start > a->shape[i])
            start = a->shape[i];
        if (end < 0)
            end = 0;
        if (end > a->shape[i])
            end = a->shape[i];

        int len = (end - start + step - 1) / step;
        if (len < 0)
            len = 0;
        new_shape[i] = len;

        storage_offset += (size_t)start * strides[i];
    }

    // Create view
    // New strides = old strides * step
    size_t* new_strides = malloc((size_t)a->ndim * sizeof(size_t));
    if (!new_strides) {
        free(new_shape);
        if (free_strides)
            free(strides);
        return NULL;
    }

    for (int i = 0; i < a->ndim; i++) {
        int step       = params->step ? params->step[i] : 1;
        new_strides[i] = strides[i] * (size_t)step;
    }

    Tensor* result = tensor_as_strided(a, new_shape, a->ndim, new_strides, storage_offset);

    free(new_shape);
    free(new_strides);
    if (free_strides)
        free(strides);

    if (result) {
        CMLIR_t ir = cml_ir_get_or_create_context();
        if (ir) {
            // Deep copy params
            SliceParams* new_params = malloc(sizeof(SliceParams));
            if (new_params) {
                new_params->num_dims = params->num_dims;
                new_params->start    = malloc((size_t)params->num_dims * sizeof(int));
                new_params->end      = malloc((size_t)params->num_dims * sizeof(int));
                new_params->step     = malloc((size_t)params->num_dims * sizeof(int));

                if (new_params->start && new_params->end && new_params->step) {
                    memcpy(new_params->start, params->start,
                           (size_t)params->num_dims * sizeof(int));
                    memcpy(new_params->end, params->end, (size_t)params->num_dims * sizeof(int));
                    if (params->step) {
                        memcpy(new_params->step, params->step,
                               (size_t)params->num_dims * sizeof(int));
                    } else {
                        for (int i = 0; i < params->num_dims; i++)
                            new_params->step[i] = 1;
                    }

                    Tensor* inputs[] = {a};
                    if (cml_ir_add_uop(ir, UOP_SLICE, inputs, 1, new_params) == 0) {
                        struct IRNode* node = cml_ir_get_tail(ir);
                        node->output_shape  = tensor_shape_copy(result->shape, result->ndim);
                        node->output_ndim   = result->ndim;
                        if (a->requires_grad) {
                            node->requires_grad       = true;
                            node->needs_input_grad[0] = true;
                        }
                        result->ir_node    = node;
                        result->ir_context = ir;
                        node->output       = result;
                    } else {
                        free(new_params->start);
                        free(new_params->end);
                        free(new_params->step);
                        free(new_params);
                    }
                } else {
                    if (new_params->start)
                        free(new_params->start);
                    if (new_params->end)
                        free(new_params->end);
                    if (new_params->step)
                        free(new_params->step);
                    free(new_params);
                }
            }
        }
    }

    return result;
}

Tensor* uop_matmul(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to uop_matmul");
        return NULL;
    }

    extern Tensor* tensor_matmul(Tensor*, Tensor*);
    return tensor_matmul(a, b);
}

static void conv2d_params_free_local(Conv2DParams* p) {
    if (!p)
        return;
    if (p->kernel_size)
        free(p->kernel_size);
    if (p->stride)
        free(p->stride);
    if (p->padding)
        free(p->padding);
    if (p->dilation)
        free(p->dilation);
    free(p);
}

Tensor* uop_conv2d(Tensor* input, Tensor* weight, Tensor* bias, Conv2DParams* params) {
    if (!input || !weight) {
        LOG_ERROR("NULL tensor input to uop_conv2d");
        error_stack_push(CM_INVALID_ARGUMENT, "uop_conv2d: NULL tensor input", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    if (input->ndim != 4) {
        LOG_ERROR("Conv2D expects 4D input [batch, in_channels, height, width], got %dD",
                  input->ndim);
        error_stack_push(CM_INVALID_ARGUMENT, "uop_conv2d: invalid input rank", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    if (weight->ndim != 4) {
        LOG_ERROR("Conv2D weight must be 4D [out_channels, in_channels, kernel_h, kernel_w], got "
                  "%dD",
                  weight->ndim);
        error_stack_push(CM_INVALID_ARGUMENT, "uop_conv2d: invalid weight rank", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    int batch       = input->shape[0];
    int in_channels = input->shape[1];
    int in_height   = input->shape[2];
    int in_width    = input->shape[3];

    int out_channels       = weight->shape[0];
    int weight_in_channels = weight->shape[1];
    int kernel_h           = weight->shape[2];
    int kernel_w           = weight->shape[3];

    if (in_channels != weight_in_channels) {
        LOG_ERROR("Conv2D: input channels (%d) doesn't match weight in_channels (%d)", in_channels,
                  weight_in_channels);
        error_stack_push(CM_INVALID_ARGUMENT, "uop_conv2d: channel mismatch", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    int stride_h   = params && params->stride ? params->stride[0] : 1;
    int stride_w   = params && params->stride ? params->stride[1] : 1;
    int padding_h  = params && params->padding ? params->padding[0] : 0;
    int padding_w  = params && params->padding ? params->padding[1] : 0;
    int dilation_h = params && params->dilation ? params->dilation[0] : 1;
    int dilation_w = params && params->dilation ? params->dilation[1] : 1;

    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width  = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    if (out_height <= 0 || out_width <= 0) {
        LOG_ERROR("Conv2D: invalid output dimensions (%d x %d)", out_height, out_width);
        error_stack_push(CM_INVALID_ARGUMENT, "uop_conv2d: invalid output dimensions", __FILE__,
                         __LINE__, __func__);
        return NULL;
    }

    if (bias) {
        if (bias->ndim != 1 || bias->shape[0] != out_channels) {
            LOG_ERROR("Conv2D bias must be 1D with shape [out_channels]");
            error_stack_push(CM_INVALID_ARGUMENT, "uop_conv2d: invalid bias shape", __FILE__,
                             __LINE__, __func__);
            return NULL;
        }
    }

    Conv2DParams* params_copy = calloc(1, sizeof(Conv2DParams));
    if (!params_copy) {
        error_stack_push(CM_OPERATION_FAILED, "uop_conv2d: params alloc failed", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    params_copy->kernel_size = malloc(2 * sizeof(int));
    params_copy->stride      = malloc(2 * sizeof(int));
    params_copy->padding     = malloc(2 * sizeof(int));
    params_copy->dilation    = malloc(2 * sizeof(int));
    if (!params_copy->kernel_size || !params_copy->stride || !params_copy->padding ||
        !params_copy->dilation) {
        conv2d_params_free_local(params_copy);
        error_stack_push(CM_OPERATION_FAILED, "uop_conv2d: params alloc failed", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    params_copy->kernel_size[0] = kernel_h;
    params_copy->kernel_size[1] = kernel_w;
    params_copy->stride[0]      = stride_h;
    params_copy->stride[1]      = stride_w;
    params_copy->padding[0]     = padding_h;
    params_copy->padding[1]     = padding_w;
    params_copy->dilation[0]    = dilation_h;
    params_copy->dilation[1]    = dilation_w;
    params_copy->groups         = params ? params->groups : 1;

    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        conv2d_params_free_local(params_copy);
        return NULL;
    }

    Tensor* inputs[3] = {input, weight, bias};
    int num_inputs    = bias ? 3 : 2;

    if (cml_ir_add_uop(ir, UOP_CONV2D, inputs, num_inputs, params_copy) != 0) {
        conv2d_params_free_local(params_copy);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    if (!node) {
        return NULL;
    }

    int output_shape[4] = {batch, out_channels, out_height, out_width};
    node->output_shape  = tensor_shape_copy(output_shape, 4);
    node->output_ndim   = 4;

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_fill(int* shape, int ndim, float value) {
    if (!shape || ndim <= 0) {
        LOG_ERROR("Invalid shape for uop_fill");
        return NULL;
    }

    // UOps ONLY create IR nodes, never execute! (Lazy evaluation)
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    // Deep copy params for IR node ownership
    FillParams* params = malloc(sizeof(FillParams));
    if (!params)
        return NULL;

    params->value = value;
    params->ndim  = ndim;
    params->shape = malloc((size_t)ndim * sizeof(int));
    if (!params->shape) {
        free(params);
        return NULL;
    }
    memcpy(params->shape, shape, (size_t)ndim * sizeof(int));

    // UOP_FILL has no inputs - it's a source node
    // We pass NULL inputs with 0 count, but need special handling
    if (cml_ir_add_uop(ir, UOP_FILL, NULL, 0, params) != 0) {
        free(params->shape);
        free(params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    if (!node) {
        return NULL;
    }

    // Set output shape
    node->output_shape = tensor_shape_copy(shape, ndim);
    node->output_ndim  = ndim;

    // Fill doesn't require grad
    node->requires_grad = false;

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_gather(Tensor* input, Tensor* indices, int dim) {
    if (!input || !indices) {
        LOG_ERROR("NULL tensor input to uop_gather");
        return NULL;
    }

    // For cross-entropy: input is [N, C], indices is [N], output is [N]
    // out[i] = input[i, indices[i]]
    if (input->ndim < 1 || indices->ndim != 1) {
        LOG_ERROR("uop_gather: input must have at least 1 dim, indices must be 1D");
        return NULL;
    }

    // Handle negative dim
    int gather_dim = dim;
    if (gather_dim < 0) {
        gather_dim = input->ndim + gather_dim;
    }
    if (gather_dim < 0 || gather_dim >= input->ndim) {
        LOG_ERROR("uop_gather: invalid dimension %d for input with %d dims", dim, input->ndim);
        return NULL;
    }

    // UOps ONLY create IR nodes, never execute! (Lazy evaluation)
    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    // Deep copy params for IR node ownership
    GatherParams* params = malloc(sizeof(GatherParams));
    if (!params)
        return NULL;
    params->dim = gather_dim;

    Tensor* inputs[] = {input, indices};
    if (cml_ir_add_uop(ir, UOP_GATHER, inputs, 2, params) != 0) {
        free(params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    if (!node) {
        return NULL;
    }

    // Compute output shape: for gather along last dim with 2D input [N,C] and 1D indices [N]
    // Output is [N] (same as indices shape)
    node->output_shape = tensor_shape_copy(indices->shape, indices->ndim);
    node->output_ndim  = indices->ndim;

    // Gather supports gradient only for input (not indices)
    if (input->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
        node->needs_input_grad[1] = false; // indices don't have gradients
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_where(WhereParams* params) {
    if (!params || !params->cond || !params->a || !params->b) {
        LOG_ERROR("NULL input to uop_where");
        return NULL;
    }

    // WHERE: where(cond, a, b) = cond ? a : b
    // Implement as: cond * a + (1 - cond) * b

    // Create (1 - cond) using lazy uop_fill for ones
    Tensor* ones = uop_fill(params->cond->shape, params->cond->ndim, 1.0f);
    if (!ones)
        return NULL;

    Tensor* not_cond = uop_sub(ones, params->cond);
    // Don't free ones - it's part of the IR graph
    if (!not_cond)
        return NULL;

    // cond * a
    Tensor* term1 = uop_mul(params->cond, params->a);
    if (!term1) {
        // Don't free not_cond - it's part of the IR graph
        return NULL;
    }

    // (1 - cond) * b
    Tensor* term2 = uop_mul(not_cond, params->b);
    // Don't free not_cond - it's part of the IR graph
    if (!term2) {
        // Don't free term1 - it's part of the IR graph
        return NULL;
    }

    // cond * a + (1 - cond) * b
    Tensor* result = uop_add(term1, term2);
    // Don't free term1 and term2 - they're part of the IR graph

    return result;
}

Tensor* uop_relu(Tensor* x) {
    if (!x) {
        LOG_ERROR("NULL tensor input to uop_relu");
        return NULL;
    }

    // ReLU: max(x, 0) - create zeros lazily
    Tensor* zeros = uop_fill(x->shape, x->ndim, 0.0f);
    if (!zeros)
        return NULL;

    Tensor* result = uop_max(x, zeros);
    // Don't free zeros - it's part of the IR graph
    return result;
}

Tensor* uop_sigmoid(Tensor* x) {
    if (!x) {
        LOG_ERROR("NULL tensor input to uop_sigmoid");
        return NULL;
    }

    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_SIGMOID, inputs, 1, NULL) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    if (!node)
        return NULL;

    // Sigmoid preserves shape
    node->output_shape = (int*)malloc(x->ndim * sizeof(int));
    if (!node->output_shape)
        return NULL;
    memcpy(node->output_shape, x->shape, x->ndim * sizeof(int));
    node->output_ndim = x->ndim;

    if (x->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_tanh(Tensor* x) {
    if (!x) {
        LOG_ERROR("NULL tensor input to uop_tanh");
        return NULL;
    }

    CMLIR_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_TANH, inputs, 1, NULL) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    if (!node)
        return NULL;

    // Tanh preserves shape
    node->output_shape = (int*)malloc(x->ndim * sizeof(int));
    if (!node->output_shape)
        return NULL;
    memcpy(node->output_shape, x->shape, x->ndim * sizeof(int));
    node->output_ndim = x->ndim;

    if (x->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_gelu(Tensor* x) {
    if (!x) {
        LOG_ERROR("NULL tensor input to uop_gelu");
        return NULL;
    }

    // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    // Approximate: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    // Constants (all created lazily with uop_fill)
    const float sqrt_2_pi = 0.7978845608f;
    const float coeff     = 0.044715f;

    // x^3 (lazy)
    Tensor* x2 = uop_mul(x, x);
    if (!x2)
        return NULL;
    Tensor* x3 = uop_mul(x2, x);
    if (!x3)
        return NULL;

    // 0.044715 * x^3 (lazy constant via uop_fill)
    Tensor* coeff_tensor = uop_fill(x->shape, x->ndim, coeff);
    if (!coeff_tensor)
        return NULL;

    Tensor* scaled_x3 = uop_mul(coeff_tensor, x3);
    if (!scaled_x3)
        return NULL;

    // x + 0.044715 * x^3 (lazy)
    Tensor* arg = uop_add(x, scaled_x3);
    if (!arg)
        return NULL;

    // sqrt(2/π) * (x + 0.044715 * x^3) (lazy constant via uop_fill)
    Tensor* sqrt_tensor = uop_fill(x->shape, x->ndim, sqrt_2_pi);
    if (!sqrt_tensor)
        return NULL;

    Tensor* scaled_arg = uop_mul(sqrt_tensor, arg);
    if (!scaled_arg)
        return NULL;

    // tanh(...) (lazy)
    Tensor* tanh_result = uop_tanh(scaled_arg);
    if (!tanh_result)
        return NULL;

    // 1 + tanh(...) (lazy constant via uop_fill)
    Tensor* ones = uop_fill(x->shape, x->ndim, 1.0f);
    if (!ones)
        return NULL;

    Tensor* one_plus_tanh = uop_add(ones, tanh_result);
    if (!one_plus_tanh)
        return NULL;

    // 0.5 * (1 + tanh(...)) (lazy constant via uop_fill)
    Tensor* half_tensor = uop_fill(x->shape, x->ndim, 0.5f);
    if (!half_tensor)
        return NULL;

    Tensor* scaled = uop_mul(half_tensor, one_plus_tanh);
    if (!scaled)
        return NULL;

    // x * 0.5 * (1 + tanh(...)) (lazy)
    Tensor* result = uop_mul(x, scaled);
    // All intermediate tensors are part of the IR graph

    return result;
}

Tensor* uop_softmax(Tensor* x, int dim) {
    if (!x) {
        LOG_ERROR("NULL tensor input to uop_softmax");
        return NULL;
    }

    // Softmax: exp(x - max(x, dim)) / sum(exp(x - max(x, dim)), dim)

    // max(x, dim)
    ReduceParams max_params = {0};
    max_params.dims         = &dim;
    max_params.num_dims     = 1;
    max_params.keepdim      = true;

    Tensor* max_x = uop_max_reduce(x, &max_params);
    if (!max_x)
        return NULL;

    // x - max(x, dim)
    Tensor* x_sub_max = uop_sub(x, max_x);
    // Don't free max_x - it's part of the IR graph
    if (!x_sub_max)
        return NULL;

    // exp(x - max(x, dim))
    Tensor* exp_x = uop_exp(x_sub_max);
    // Don't free x_sub_max - it's part of the IR graph
    if (!exp_x)
        return NULL;

    // sum(exp(...), dim)
    ReduceParams sum_params = {0};
    sum_params.dims         = &dim;
    sum_params.num_dims     = 1;
    sum_params.keepdim      = true;

    Tensor* sum_exp = uop_sum(exp_x, &sum_params);
    if (!sum_exp) {
        // Don't free exp_x - it's part of the IR graph
        return NULL;
    }

    // exp(...) / sum(...)
    Tensor* result = uop_div(exp_x, sum_exp);
    // Don't free exp_x and sum_exp - they're part of the IR graph

    return result;
}

Tensor* uop_leaky_relu(Tensor* x, float negative_slope) {
    if (!x) {
        LOG_ERROR("NULL tensor input to uop_leaky_relu");
        return NULL;
    }

    // Leaky ReLU: max(x, alpha * x) where alpha = negative_slope
    // For x > 0: max(x, alpha*x) = x (since alpha < 1 typically)
    // For x <= 0: max(x, alpha*x) = alpha*x

    // Create alpha tensor lazily using uop_fill
    Tensor* alpha_tensor = uop_fill(x->shape, x->ndim, negative_slope);
    if (!alpha_tensor)
        return NULL;

    // alpha * x (lazy)
    Tensor* alpha_x = uop_mul(alpha_tensor, x);
    // Don't free alpha_tensor - it's part of the IR graph
    if (!alpha_x)
        return NULL;

    // max(x, alpha * x) (lazy)
    Tensor* result = uop_max(x, alpha_x);
    // Don't free alpha_x - it's part of the IR graph

    return result;
}

int uop_execute(UOp* uop) {
    if (!uop || !uop->inputs) {
        LOG_ERROR("Invalid uop");
        return -1;
    }

    Tensor* result = uop_create_and_execute(uop->type, uop->inputs, uop->num_inputs, uop->params);
    if (!result) {
        return -1;
    }

    uop->output = result;
    return 0;
}

Tensor* uop_create_and_execute(UOpType type, Tensor** inputs, int num_inputs, void* params) {
    // Note: params is used for WHERE, SUM, MEAN, MAX_REDUCE, CONV2D operations
    // For other operations, params may be NULL or unused (intentional)
    (void)params; // Suppress warning for operations that don't use params (intentional)

    if (!inputs || num_inputs <= 0) {
        LOG_ERROR("Invalid inputs for uop");
        return NULL;
    }

    switch (type) {
    case UOP_ADD:
        if (num_inputs >= 2)
            return uop_add(inputs[0], inputs[1]);
        break;
    case UOP_SUB:
        if (num_inputs >= 2)
            return uop_sub(inputs[0], inputs[1]);
        break;
    case UOP_MUL:
        if (num_inputs >= 2)
            return uop_mul(inputs[0], inputs[1]);
        break;
    case UOP_DIV:
        if (num_inputs >= 2)
            return uop_div(inputs[0], inputs[1]);
        break;
    case UOP_MAX:
        if (num_inputs >= 2)
            return uop_max(inputs[0], inputs[1]);
        break;
    case UOP_NEG:
        if (num_inputs >= 1)
            return uop_neg(inputs[0]);
        break;
    case UOP_EXP:
        if (num_inputs >= 1)
            return uop_exp(inputs[0]);
        break;
    case UOP_LOG:
        if (num_inputs >= 1)
            return uop_log(inputs[0]);
        break;
    case UOP_SQRT:
        if (num_inputs >= 1)
            return uop_sqrt(inputs[0]);
        break;
    case UOP_RECIP:
        if (num_inputs >= 1)
            return uop_recip(inputs[0]);
        break;
    case UOP_ABS:
        if (num_inputs >= 1)
            return uop_abs(inputs[0]);
        break;
    case UOP_SIN:
        if (num_inputs >= 1)
            return uop_sin(inputs[0]);
        break;
    case UOP_COS:
        if (num_inputs >= 1)
            return uop_cos(inputs[0]);
        break;
    case UOP_TAN:
        if (num_inputs >= 1)
            return uop_tan(inputs[0]);
        break;
    case UOP_POW:
        if (num_inputs >= 2)
            return uop_pow(inputs[0], inputs[1]);
        break;
    case UOP_CMPLT:
        if (num_inputs >= 2)
            return uop_cmplt(inputs[0], inputs[1]);
        break;
    case UOP_MATMUL:
        if (num_inputs >= 2)
            return uop_matmul(inputs[0], inputs[1]);
        break;
    case UOP_WHERE:
        if (num_inputs >= 3 && params) {
            WhereParams* where_params = (WhereParams*)params;
            where_params->cond        = inputs[0];
            where_params->a           = inputs[1];
            where_params->b           = inputs[2];
            return uop_where(where_params);
        }
        break;
    case UOP_SUM:
    case UOP_MEAN:
    case UOP_MAX_REDUCE:
        if (num_inputs >= 1) {
            ReduceParams* reduce_params = (ReduceParams*)params;
            if (type == UOP_SUM)
                return uop_sum(inputs[0], reduce_params);
            if (type == UOP_MEAN)
                return uop_mean(inputs[0], reduce_params);
            if (type == UOP_MAX_REDUCE)
                return uop_max_reduce(inputs[0], reduce_params);
        }
        break;
    case UOP_RESHAPE:
    case UOP_PERMUTE:
    case UOP_EXPAND:
    case UOP_STRIDE:
    case UOP_SLICE:
        // Movement ops require params, handled by specific uop functions
        LOG_ERROR("Movement ops (RESHAPE, PERMUTE, EXPAND, STRIDE, SLICE) must be called directly, "
                  "not via uop_create_and_execute");
        break;
    case UOP_CONV2D:
        if (num_inputs >= 2) {
            Tensor* bias              = num_inputs >= 3 ? inputs[2] : NULL;
            Conv2DParams* conv_params = (Conv2DParams*)params;
            return uop_conv2d(inputs[0], inputs[1], bias, conv_params);
        }
        break;
    case UOP_COUNT:
        break;
    default:
        LOG_ERROR("UOp type %d not implemented", type);
        break;
    }

    return NULL;
}
