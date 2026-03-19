#include "ops/uops.h"
#include "ops/winograd.h"
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

static Tensor* uop_binary(Tensor* a, Tensor* b, UOpType type) {
    if (!a || !b)
        return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;
    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, type, inputs, 2, NULL) != 0)
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

Tensor* uop_add(Tensor* a, Tensor* b) { return uop_binary(a, b, UOP_ADD); }
Tensor* uop_sub(Tensor* a, Tensor* b) { return uop_binary(a, b, UOP_SUB); }
Tensor* uop_mul(Tensor* a, Tensor* b) { return uop_binary(a, b, UOP_MUL); }
Tensor* uop_div(Tensor* a, Tensor* b) { return uop_binary(a, b, UOP_DIV); }

Tensor* uop_max(Tensor* a, Tensor* b) {
    if (!a || !b) {
        LOG_ERROR("NULL tensor input to uop_max");
        return NULL;
    }

    if (!tensor_can_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim)) {
        LOG_ERROR("Shapes cannot be broadcast for uop_max");
        return NULL;
    }

    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape)
        return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
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

    if (!tensor_can_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim)) {
        LOG_ERROR("Shapes cannot be broadcast for uop_cmplt");
        return NULL;
    }

    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape)
        return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
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

    if (a->requires_grad || b->requires_grad) {
        node->requires_grad       = false;
        node->needs_input_grad[0] = false;
        node->needs_input_grad[1] = false;
    }

    return tensor_from_ir_node(node, ir);
}

static Tensor* uop_unary(Tensor* a, UOpType type) {
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

Tensor* uop_neg(Tensor* a)  { return uop_unary(a, UOP_NEG); }
Tensor* uop_exp(Tensor* a)  { return uop_unary(a, UOP_EXP); }
Tensor* uop_log(Tensor* a)  { return uop_unary(a, UOP_LOG); }
Tensor* uop_sqrt(Tensor* a) { return uop_unary(a, UOP_SQRT); }

Tensor* uop_recip(Tensor* a) {
    if (!a) {
        LOG_ERROR("NULL tensor input to uop_recip");
        return NULL;
    }

    Tensor* ones = uop_fill(a->shape, a->ndim, 1.0f);
    if (!ones)
        return NULL;
    return uop_div(ones, a);
}

Tensor* uop_abs(Tensor* a) { return uop_unary(a, UOP_ABS); }
Tensor* uop_sin(Tensor* a) { return uop_unary(a, UOP_SIN); }
Tensor* uop_cos(Tensor* a) { return uop_unary(a, UOP_COS); }
Tensor* uop_tan(Tensor* a) { return uop_unary(a, UOP_TAN); }

Tensor* uop_pow(Tensor* a, Tensor* b) { return uop_binary(a, b, UOP_POW); }

static Tensor* uop_reduce(Tensor* a, ReduceParams* params, UOpType type) {
    if (!a)
        return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

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
    if (cml_ir_add_uop(ir, type, inputs, 1, new_params) != 0) {
        free(new_params->dims);
        free(new_params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);

    int dim      = params && params->dims && params->num_dims > 0 ? params->dims[0] : -1;
    bool keepdim = params ? params->keepdim : false;

    int* out_shape = NULL;
    int out_ndim   = 0;

    if (dim < 0 || dim >= a->ndim) {
        out_ndim  = 1;
        out_shape = malloc(sizeof(int));
        if (!out_shape)
            return NULL;
        out_shape[0] = 1;
    } else {
        out_ndim = keepdim ? a->ndim : (a->ndim - 1);
        if (out_ndim == 0)
            out_ndim = 1;
        out_shape = malloc((size_t)out_ndim * sizeof(int));
        if (!out_shape)
            return NULL;

        int out_idx = 0;
        if (keepdim) {
            for (int i = 0; i < a->ndim; i++)
                out_shape[i] = (i == dim) ? 1 : a->shape[i];
        } else {
            for (int i = 0; i < a->ndim; i++)
                if (i != dim)
                    out_shape[out_idx++] = a->shape[i];
        }
    }

    node->output_shape = out_shape;
    node->output_ndim  = out_ndim;

    if (a->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_sum(Tensor* a, ReduceParams* params)        { return uop_reduce(a, params, UOP_SUM); }
Tensor* uop_max_reduce(Tensor* a, ReduceParams* params) { return uop_reduce(a, params, UOP_MAX_REDUCE); }
Tensor* uop_mean(Tensor* a, ReduceParams* params)       { return uop_reduce(a, params, UOP_MEAN); }

Tensor* uop_reshape(Tensor* a, ReshapeParams* params) {
    if (!a || !params) {
        LOG_ERROR("NULL input to uop_reshape");
        return NULL;
    }

    Tensor* result = tensor_reshape(a, params->new_shape, params->new_ndim);

    if (result) {
        CMLGraph_t ir = cml_ir_get_or_create_context();
        if (ir) {
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

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

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

    int* out_shape = malloc((size_t)a->ndim * sizeof(int));
    if (!out_shape)
        return NULL;

    for (int i = 0; i < a->ndim; i++) {
        out_shape[i] = a->shape[params->perm[i]];
    }

    node->output_shape = out_shape;
    node->output_ndim  = a->ndim;

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

    if (params->new_ndim < a->ndim) {
        LOG_ERROR("Expand: new_ndim (%d) must be >= tensor ndim (%d)", params->new_ndim, a->ndim);
        return NULL;
    }

    int* broadcast_shape = malloc((size_t)params->new_ndim * sizeof(int));
    if (!broadcast_shape)
        return NULL;

    int prepend = params->new_ndim - a->ndim;
    for (int i = 0; i < prepend; i++) {
        broadcast_shape[i] = 1;
    }

    for (int i = 0; i < a->ndim; i++) {
        broadcast_shape[prepend + i] = a->shape[i];
    }

    for (int i = 0; i < params->new_ndim; i++) {
        int orig_dim = i >= prepend ? a->shape[i - prepend] : 1;
        if (orig_dim != params->new_shape[i] && orig_dim != 1 && params->new_shape[i] != 1) {
            free(broadcast_shape);
            LOG_ERROR("Expand: cannot broadcast dimension %d: %d -> %d", i, orig_dim,
                      params->new_shape[i]);
            return NULL;
        }
    }

    size_t* new_strides = malloc((size_t)params->new_ndim * sizeof(size_t));
    if (!new_strides) {
        free(broadcast_shape);
        return NULL;
    }

    size_t* orig_strides = a->strides ? a->strides : compute_contiguous_strides(a->shape, a->ndim);
    if (!orig_strides) {
        free(broadcast_shape);
        free(new_strides);
        return NULL;
    }

    new_strides[params->new_ndim - 1] = 1;
    for (int i = params->new_ndim - 2; i >= 0; i--) {
        if (i < prepend) {
            new_strides[i] = 0;
        } else {
            int orig_idx = i - prepend;
            if (broadcast_shape[i] == 1 && params->new_shape[i] != 1) {
                new_strides[i] = 0;
            } else {
                new_strides[i] = orig_strides[orig_idx] *
                                 ((size_t)params->new_shape[i] / (size_t)broadcast_shape[i]);
            }
        }
    }

    if (!a->strides) {
        free(orig_strides);
    }

    Tensor* result =
        tensor_as_strided(a, params->new_shape, params->new_ndim, new_strides, a->storage_offset);

    free(broadcast_shape);
    free(new_strides);

    if (result) {
        CMLGraph_t ir = cml_ir_get_or_create_context();
        if (ir) {
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

    Tensor* result =
        tensor_as_strided(a, a->shape, a->ndim, params->new_strides, a->storage_offset);

    if (result) {
        CMLGraph_t ir = cml_ir_get_or_create_context();
        if (ir) {
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

        if (start < 0)
            start += a->shape[i];
        if (end < 0)
            end += a->shape[i];

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
        CMLGraph_t ir = cml_ir_get_or_create_context();
        if (ir) {
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

    params_copy->use_winograd = winograd_applicable(
        kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w);

    CMLGraph_t ir = cml_ir_get_or_create_context();
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

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

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

    if (cml_ir_add_uop(ir, UOP_FILL, NULL, 0, params) != 0) {
        free(params->shape);
        free(params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    if (!node) {
        return NULL;
    }

    node->output_shape = tensor_shape_copy(shape, ndim);
    node->output_ndim  = ndim;
    node->requires_grad = false;

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_gather(Tensor* input, Tensor* indices, int dim) {
    if (!input || !indices) {
        LOG_ERROR("NULL tensor input to uop_gather");
        return NULL;
    }

    if (input->ndim < 1 || indices->ndim != 1) {
        LOG_ERROR("uop_gather: input must have at least 1 dim, indices must be 1D");
        return NULL;
    }

    int gather_dim = dim;
    if (gather_dim < 0) {
        gather_dim = input->ndim + gather_dim;
    }
    if (gather_dim < 0 || gather_dim >= input->ndim) {
        LOG_ERROR("uop_gather: invalid dimension %d for input with %d dims", dim, input->ndim);
        return NULL;
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

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

    node->output_shape = tensor_shape_copy(indices->shape, indices->ndim);
    node->output_ndim  = indices->ndim;

    if (input->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
        node->needs_input_grad[1] = false;
    }

    return tensor_from_ir_node(node, ir);
}

Tensor* uop_where(WhereParams* params) {
    if (!params || !params->cond || !params->a || !params->b) {
        LOG_ERROR("NULL input to uop_where");
        return NULL;
    }

    Tensor* ones = uop_fill(params->cond->shape, params->cond->ndim, 1.0f);
    if (!ones)
        return NULL;

    Tensor* not_cond = uop_sub(ones, params->cond);
    if (!not_cond)
        return NULL;

    Tensor* term1 = uop_mul(params->cond, params->a);
    if (!term1)
        return NULL;

    Tensor* term2 = uop_mul(not_cond, params->b);
    if (!term2)
        return NULL;

    return uop_add(term1, term2);
}

Tensor* uop_relu(Tensor* x) {
    if (!x) {
        LOG_ERROR("NULL tensor input to uop_relu");
        return NULL;
    }

    Tensor* zeros = uop_fill(x->shape, x->ndim, 0.0f);
    if (!zeros)
        return NULL;
    return uop_max(x, zeros);
}

Tensor* uop_sigmoid(Tensor* x) {
    if (!x) {
        LOG_ERROR("NULL tensor input to uop_sigmoid");
        return NULL;
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_SIGMOID, inputs, 1, NULL) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    if (!node)
        return NULL;

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

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir)
        return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_TANH, inputs, 1, NULL) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    if (!node)
        return NULL;

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

    Tensor* x2 = uop_mul(x, x);
    if (!x2)
        return NULL;
    Tensor* x3 = uop_mul(x2, x);
    if (!x3)
        return NULL;

    Tensor* coeff_tensor = uop_fill(x->shape, x->ndim, coeff);
    if (!coeff_tensor)
        return NULL;
    Tensor* scaled_x3 = uop_mul(coeff_tensor, x3);
    if (!scaled_x3)
        return NULL;

    Tensor* arg = uop_add(x, scaled_x3);
    if (!arg)
        return NULL;

    Tensor* sqrt_tensor = uop_fill(x->shape, x->ndim, sqrt_2_pi);
    if (!sqrt_tensor)
        return NULL;
    Tensor* scaled_arg = uop_mul(sqrt_tensor, arg);
    if (!scaled_arg)
        return NULL;

    Tensor* tanh_result = uop_tanh(scaled_arg);
    if (!tanh_result)
        return NULL;

    Tensor* ones = uop_fill(x->shape, x->ndim, 1.0f);
    if (!ones)
        return NULL;
    Tensor* one_plus_tanh = uop_add(ones, tanh_result);
    if (!one_plus_tanh)
        return NULL;

    Tensor* half_tensor = uop_fill(x->shape, x->ndim, 0.5f);
    if (!half_tensor)
        return NULL;
    Tensor* scaled = uop_mul(half_tensor, one_plus_tanh);
    if (!scaled)
        return NULL;

    return uop_mul(x, scaled);
}

Tensor* uop_softmax(Tensor* x, int dim) {
    if (!x) {
        LOG_ERROR("NULL tensor input to uop_softmax");
        return NULL;
    }

    ReduceParams max_params = {0};
    max_params.dims         = &dim;
    max_params.num_dims     = 1;
    max_params.keepdim      = true;

    Tensor* max_x = uop_max_reduce(x, &max_params);
    if (!max_x)
        return NULL;

    Tensor* x_sub_max = uop_sub(x, max_x);
    if (!x_sub_max)
        return NULL;

    Tensor* exp_x = uop_exp(x_sub_max);
    if (!exp_x)
        return NULL;

    ReduceParams sum_params = {0};
    sum_params.dims         = &dim;
    sum_params.num_dims     = 1;
    sum_params.keepdim      = true;

    Tensor* sum_exp = uop_sum(exp_x, &sum_params);
    if (!sum_exp)
        return NULL;

    return uop_div(exp_x, sum_exp);
}

Tensor* uop_leaky_relu(Tensor* x, float negative_slope) {
    if (!x) {
        LOG_ERROR("NULL tensor input to uop_leaky_relu");
        return NULL;
    }

    Tensor* alpha_tensor = uop_fill(x->shape, x->ndim, negative_slope);
    if (!alpha_tensor)
        return NULL;

    Tensor* alpha_x = uop_mul(alpha_tensor, x);
    if (!alpha_x)
        return NULL;

    return uop_max(x, alpha_x);
}

#define DEFINE_SIMPLE_UNARY_UOP(name, uop_type)                                                    \
    Tensor* name(Tensor* a) {                                                                      \
        if (!a) return NULL;                                                                       \
        CMLGraph_t ir = cml_ir_get_or_create_context();                                            \
        if (!ir) return NULL;                                                                      \
        Tensor* inputs[] = {a};                                                                    \
        if (cml_ir_add_uop(ir, uop_type, inputs, 1, NULL) != 0) return NULL;                      \
        struct IRNode* node = cml_ir_get_tail(ir);                                                 \
        node->output_shape = tensor_shape_copy(a->shape, a->ndim);                                 \
        node->output_ndim = a->ndim;                                                               \
        if (a->requires_grad) {                                                                    \
            node->requires_grad = true;                                                            \
            node->needs_input_grad[0] = true;                                                      \
        }                                                                                          \
        return tensor_from_ir_node(node, ir);                                                      \
    }

DEFINE_SIMPLE_UNARY_UOP(uop_sign, UOP_SIGN)
DEFINE_SIMPLE_UNARY_UOP(uop_floor, UOP_FLOOR)
DEFINE_SIMPLE_UNARY_UOP(uop_ceil, UOP_CEIL)
DEFINE_SIMPLE_UNARY_UOP(uop_round, UOP_ROUND)
DEFINE_SIMPLE_UNARY_UOP(uop_log2, UOP_LOG2)
DEFINE_SIMPLE_UNARY_UOP(uop_exp2, UOP_EXP2)
DEFINE_SIMPLE_UNARY_UOP(uop_asin, UOP_ASIN)
DEFINE_SIMPLE_UNARY_UOP(uop_acos, UOP_ACOS)
DEFINE_SIMPLE_UNARY_UOP(uop_atan, UOP_ATAN)
DEFINE_SIMPLE_UNARY_UOP(uop_square, UOP_SQUARE)
DEFINE_SIMPLE_UNARY_UOP(uop_rsqrt, UOP_RSQRT)
DEFINE_SIMPLE_UNARY_UOP(uop_erf, UOP_ERF)

#undef DEFINE_SIMPLE_UNARY_UOP

Tensor* uop_clamp(Tensor* a, float min_val, float max_val) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    ClampParams* params = malloc(sizeof(ClampParams));
    if (!params) return NULL;
    params->min_val = min_val;
    params->max_val = max_val;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_CLAMP, inputs, 1, params) != 0) {
        free(params);
        return NULL;
    }
    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_prod(Tensor* a, ReduceParams* params) {
    return uop_reduce(a, params, UOP_PROD);
}

Tensor* uop_argmax(Tensor* a, ReduceParams* params) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    ReduceParams* new_params = malloc(sizeof(ReduceParams));
    if (!new_params) return NULL;
    new_params->keepdim = params ? params->keepdim : false;
    new_params->num_dims = params ? params->num_dims : 0;
    if (params && params->dims && params->num_dims > 0) {
        new_params->dims = malloc((size_t)params->num_dims * sizeof(int));
        if (!new_params->dims) { free(new_params); return NULL; }
        memcpy(new_params->dims, params->dims, (size_t)params->num_dims * sizeof(int));
    } else {
        new_params->dims = NULL;
    }

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_ARGMAX, inputs, 1, new_params) != 0) {
        if (new_params->dims) free(new_params->dims);
        free(new_params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int dim = params && params->dims && params->num_dims > 0 ? params->dims[0] : -1;
    bool keepdim = params ? params->keepdim : false;
    int* out_shape = NULL;
    int out_ndim = 0;

    if (dim < 0 || dim >= a->ndim) {
        out_ndim = 1;
        out_shape = malloc(sizeof(int));
        if (!out_shape) return NULL;
        out_shape[0] = 1;
    } else {
        out_ndim = keepdim ? a->ndim : (a->ndim - 1);
        if (out_ndim == 0) out_ndim = 1;
        out_shape = malloc((size_t)out_ndim * sizeof(int));
        if (!out_shape) return NULL;
        int out_idx = 0;
        if (keepdim) {
            for (int i = 0; i < a->ndim; i++)
                out_shape[i] = (i == dim) ? 1 : a->shape[i];
        } else {
            for (int i = 0; i < a->ndim; i++)
                if (i != dim) out_shape[out_idx++] = a->shape[i];
        }
    }

    node->output_shape = out_shape;
    node->output_ndim = out_ndim;
    node->requires_grad = false;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_argmin(Tensor* a, ReduceParams* params) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    ReduceParams* new_params = malloc(sizeof(ReduceParams));
    if (!new_params) return NULL;
    new_params->keepdim = params ? params->keepdim : false;
    new_params->num_dims = params ? params->num_dims : 0;
    if (params && params->dims && params->num_dims > 0) {
        new_params->dims = malloc((size_t)params->num_dims * sizeof(int));
        if (!new_params->dims) { free(new_params); return NULL; }
        memcpy(new_params->dims, params->dims, (size_t)params->num_dims * sizeof(int));
    } else {
        new_params->dims = NULL;
    }

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_ARGMIN, inputs, 1, new_params) != 0) {
        if (new_params->dims) free(new_params->dims);
        free(new_params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int dim = params && params->dims && params->num_dims > 0 ? params->dims[0] : -1;
    bool keepdim = params ? params->keepdim : false;
    int* out_shape = NULL;
    int out_ndim = 0;

    if (dim < 0 || dim >= a->ndim) {
        out_ndim = 1;
        out_shape = malloc(sizeof(int));
        if (!out_shape) return NULL;
        out_shape[0] = 1;
    } else {
        out_ndim = keepdim ? a->ndim : (a->ndim - 1);
        if (out_ndim == 0) out_ndim = 1;
        out_shape = malloc((size_t)out_ndim * sizeof(int));
        if (!out_shape) return NULL;
        int out_idx = 0;
        if (keepdim) {
            for (int i = 0; i < a->ndim; i++)
                out_shape[i] = (i == dim) ? 1 : a->shape[i];
        } else {
            for (int i = 0; i < a->ndim; i++)
                if (i != dim) out_shape[out_idx++] = a->shape[i];
        }
    }

    node->output_shape = out_shape;
    node->output_ndim = out_ndim;
    node->requires_grad = false;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_cumsum(Tensor* a, int dim) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    CumsumParams* params = malloc(sizeof(CumsumParams));
    if (!params) return NULL;
    params->dim = dim < 0 ? a->ndim + dim : dim;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_CUMSUM, inputs, 1, params) != 0) {
        free(params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_triu(Tensor* a, int diagonal) {
    if (!a || a->ndim < 2) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    TriParams* params = malloc(sizeof(TriParams));
    if (!params) return NULL;
    params->diagonal = diagonal;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_TRIU, inputs, 1, params) != 0) {
        free(params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_tril(Tensor* a, int diagonal) {
    if (!a || a->ndim < 2) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    TriParams* params = malloc(sizeof(TriParams));
    if (!params) return NULL;
    params->diagonal = diagonal;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_TRIL, inputs, 1, params) != 0) {
        free(params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_pad(Tensor* a, int* pad_widths, int num_dims, float value) {
    if (!a || !pad_widths || num_dims != a->ndim) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    PadParams* params = malloc(sizeof(PadParams));
    if (!params) return NULL;
    params->num_dims = num_dims;
    params->value = value;
    params->mode = PAD_CONSTANT;
    params->pad_widths = malloc((size_t)(num_dims * 2) * sizeof(int));
    if (!params->pad_widths) { free(params); return NULL; }
    memcpy(params->pad_widths, pad_widths, (size_t)(num_dims * 2) * sizeof(int));

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_PAD, inputs, 1, params) != 0) {
        free(params->pad_widths);
        free(params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int* out_shape = malloc((size_t)a->ndim * sizeof(int));
    if (!out_shape) return NULL;
    for (int i = 0; i < a->ndim; i++)
        out_shape[i] = a->shape[i] + pad_widths[i * 2] + pad_widths[i * 2 + 1];
    node->output_shape = out_shape;
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_sort(Tensor* a, int dim, bool descending) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    SortParams* params = malloc(sizeof(SortParams));
    if (!params) return NULL;
    params->dim = dim < 0 ? a->ndim + dim : dim;
    params->descending = descending;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_SORT, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_argsort(Tensor* a, int dim, bool descending) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    SortParams* params = malloc(sizeof(SortParams));
    if (!params) return NULL;
    params->dim = dim < 0 ? a->ndim + dim : dim;
    params->descending = descending;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_ARGSORT, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_topk(Tensor* a, int k, int dim, bool largest, Tensor** indices_out) {
    if (!a || k <= 0) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    int resolved_dim = dim < 0 ? a->ndim + dim : dim;

    TopkParams* params = malloc(sizeof(TopkParams));
    if (!params) return NULL;
    params->k = k;
    params->dim = resolved_dim;
    params->largest = largest;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_TOPK, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int* out_shape;
    int out_ndim;
    if (a->ndim == 1) {
        int shape[] = {k};
        out_shape = tensor_shape_copy(shape, 1);
        out_ndim = 1;
    } else {
        out_shape = tensor_shape_copy(a->shape, a->ndim);
        out_shape[resolved_dim] = k;
        out_ndim = a->ndim;
    }
    node->output_shape = out_shape;
    node->output_ndim = out_ndim;

    Tensor* values = tensor_from_ir_node(node, ir);

    if (indices_out) {
        SortParams* sort_params = malloc(sizeof(SortParams));
        if (sort_params) {
            sort_params->dim = resolved_dim;
            sort_params->descending = largest;

            Tensor* sort_inputs[] = {a};
            if (cml_ir_add_uop(ir, UOP_ARGSORT, sort_inputs, 1, sort_params) == 0) {
                struct IRNode* idx_node = cml_ir_get_tail(ir);
                if (a->ndim == 1) {
                    int idx_shape[] = {k};
                    idx_node->output_shape = tensor_shape_copy(idx_shape, 1);
                    idx_node->output_ndim = 1;
                } else {
                    idx_node->output_shape = tensor_shape_copy(a->shape, a->ndim);
                    idx_node->output_shape[resolved_dim] = k;
                    idx_node->output_ndim = a->ndim;
                }
                *indices_out = tensor_from_ir_node(idx_node, ir);
            } else {
                free(sort_params);
                *indices_out = NULL;
            }
        } else {
            *indices_out = NULL;
        }
    }

    return values;
}

Tensor* uop_cumprod(Tensor* a, int dim) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    CumsumParams* params = malloc(sizeof(CumsumParams));
    if (!params) return NULL;
    params->dim = dim < 0 ? a->ndim + dim : dim;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_CUMPROD, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_bitwise_and(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_BITWISE_AND, inputs, 2, NULL) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_bitwise_or(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_BITWISE_OR, inputs, 2, NULL) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_bitwise_xor(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {a, b};
    if (cml_ir_add_uop(ir, UOP_BITWISE_XOR, inputs, 2, NULL) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_bitwise_not(Tensor* a) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_BITWISE_NOT, inputs, 1, NULL) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_nonzero(Tensor* a) {
    if (!a) return NULL;
    tensor_ensure_executed(a);
    float* data = (float*)tensor_data_ptr(a);
    if (!data) return NULL;

    int count = 0;
    for (size_t i = 0; i < a->numel; i++) {
        if (data[i] != 0.0f) count++;
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_NONZERO, inputs, 1, NULL) != 0)
        return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    if (a->ndim == 1) {
        int out_shape[] = {count};
        node->output_shape = tensor_shape_copy(out_shape, 1);
        node->output_ndim = 1;
    } else {
        int out_shape[] = {count, a->ndim};
        node->output_shape = tensor_shape_copy(out_shape, 2);
        node->output_ndim = 2;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_masked_fill(Tensor* a, Tensor* mask, float value) {
    if (!a || !mask) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    MaskedFillParams* params = malloc(sizeof(MaskedFillParams));
    if (!params) return NULL;
    params->value = value;

    Tensor* inputs[] = {a, mask};
    if (cml_ir_add_uop(ir, UOP_MASKED_FILL, inputs, 2, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

#define DEFINE_SIMPLE_UNARY_UOP2(name, uop_type)                                                   \
    Tensor* name(Tensor* a) {                                                                      \
        if (!a) return NULL;                                                                       \
        CMLGraph_t ir = cml_ir_get_or_create_context();                                            \
        if (!ir) return NULL;                                                                      \
        Tensor* inputs[] = {a};                                                                    \
        if (cml_ir_add_uop(ir, uop_type, inputs, 1, NULL) != 0) return NULL;                      \
        struct IRNode* node = cml_ir_get_tail(ir);                                                 \
        node->output_shape = tensor_shape_copy(a->shape, a->ndim);                                 \
        node->output_ndim = a->ndim;                                                               \
        if (a->requires_grad) {                                                                    \
            node->requires_grad = true;                                                            \
            node->needs_input_grad[0] = true;                                                      \
        }                                                                                          \
        return tensor_from_ir_node(node, ir);                                                      \
    }

#define DEFINE_NOGRAD_UNARY_UOP(name, uop_type)                                                    \
    Tensor* name(Tensor* a) {                                                                      \
        if (!a) return NULL;                                                                       \
        CMLGraph_t ir = cml_ir_get_or_create_context();                                            \
        if (!ir) return NULL;                                                                      \
        Tensor* inputs[] = {a};                                                                    \
        if (cml_ir_add_uop(ir, uop_type, inputs, 1, NULL) != 0) return NULL;                      \
        struct IRNode* node = cml_ir_get_tail(ir);                                                 \
        node->output_shape = tensor_shape_copy(a->shape, a->ndim);                                 \
        node->output_ndim = a->ndim;                                                               \
        node->requires_grad = false;                                                               \
        return tensor_from_ir_node(node, ir);                                                      \
    }

DEFINE_SIMPLE_UNARY_UOP2(uop_log10, UOP_LOG10)
DEFINE_SIMPLE_UNARY_UOP2(uop_sinh, UOP_SINH)
DEFINE_SIMPLE_UNARY_UOP2(uop_cosh, UOP_COSH)
DEFINE_SIMPLE_UNARY_UOP2(uop_asinh, UOP_ASINH)
DEFINE_SIMPLE_UNARY_UOP2(uop_acosh, UOP_ACOSH)
DEFINE_SIMPLE_UNARY_UOP2(uop_atanh, UOP_ATANH)
DEFINE_NOGRAD_UNARY_UOP(uop_trunc, UOP_TRUNC)
DEFINE_NOGRAD_UNARY_UOP(uop_isinf, UOP_ISINF)
DEFINE_NOGRAD_UNARY_UOP(uop_isnan, UOP_ISNAN)
DEFINE_NOGRAD_UNARY_UOP(uop_isfinite, UOP_ISFINITE)
DEFINE_NOGRAD_UNARY_UOP(uop_logical_not, UOP_LOGICAL_NOT)

#undef DEFINE_SIMPLE_UNARY_UOP2
#undef DEFINE_NOGRAD_UNARY_UOP

#define DEFINE_BINARY_UOP(name, uop_type, has_grad)                                                \
    Tensor* name(Tensor* a, Tensor* b) {                                                           \
        if (!a || !b) return NULL;                                                                 \
        CMLGraph_t ir = cml_ir_get_or_create_context();                                            \
        if (!ir) return NULL;                                                                      \
        Tensor* inputs[] = {a, b};                                                                 \
        if (cml_ir_add_uop(ir, uop_type, inputs, 2, NULL) != 0) return NULL;                      \
        struct IRNode* node = cml_ir_get_tail(ir);                                                 \
        if (cml_ir_compute_broadcast_shape(node) != 0) return NULL;                                \
        if (has_grad && (a->requires_grad || b->requires_grad)) {                                  \
            node->requires_grad = true;                                                            \
            node->needs_input_grad[0] = a->requires_grad;                                         \
            node->needs_input_grad[1] = b->requires_grad;                                         \
        }                                                                                          \
        return tensor_from_ir_node(node, ir);                                                      \
    }

#define DEFINE_CMP_BINARY_UOP(name, uop_type)                                                      \
    Tensor* name(Tensor* a, Tensor* b) {                                                           \
        if (!a || !b) return NULL;                                                                 \
        CMLGraph_t ir = cml_ir_get_or_create_context();                                            \
        if (!ir) return NULL;                                                                      \
        Tensor* inputs[] = {a, b};                                                                 \
        if (cml_ir_add_uop(ir, uop_type, inputs, 2, NULL) != 0) return NULL;                      \
        struct IRNode* node = cml_ir_get_tail(ir);                                                 \
        if (cml_ir_compute_broadcast_shape(node) != 0) return NULL;                                \
        node->requires_grad = false;                                                               \
        return tensor_from_ir_node(node, ir);                                                      \
    }

DEFINE_BINARY_UOP(uop_idiv, UOP_IDIV, false)
DEFINE_BINARY_UOP(uop_mod, UOP_MOD, false)
DEFINE_BINARY_UOP(uop_minimum, UOP_MINIMUM, true)
DEFINE_BINARY_UOP(uop_copysign, UOP_COPYSIGN, false)
DEFINE_BINARY_UOP(uop_logaddexp, UOP_LOGADDEXP, true)
DEFINE_CMP_BINARY_UOP(uop_lshift, UOP_LSHIFT)
DEFINE_CMP_BINARY_UOP(uop_rshift, UOP_RSHIFT)
DEFINE_CMP_BINARY_UOP(uop_logical_and, UOP_LOGICAL_AND)
DEFINE_CMP_BINARY_UOP(uop_logical_or, UOP_LOGICAL_OR)

DEFINE_CMP_BINARY_UOP(uop_cmpeq, UOP_CMPEQ)
DEFINE_CMP_BINARY_UOP(uop_cmpne, UOP_CMPNE)
DEFINE_CMP_BINARY_UOP(uop_cmple, UOP_CMPLE)
DEFINE_CMP_BINARY_UOP(uop_cmpgt, UOP_CMPGT)
DEFINE_CMP_BINARY_UOP(uop_cmpge, UOP_CMPGE)

#undef DEFINE_BINARY_UOP
#undef DEFINE_CMP_BINARY_UOP

Tensor* uop_min_reduce(Tensor* a, ReduceParams* params) {
    return uop_reduce(a, params, UOP_MIN_REDUCE);
}

Tensor* uop_var(Tensor* a, ReduceParams* params) {
    return uop_reduce(a, params, UOP_VAR);
}

Tensor* uop_std(Tensor* a, ReduceParams* params) {
    return uop_reduce(a, params, UOP_STD);
}

Tensor* uop_any(Tensor* a, ReduceParams* params) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    ReduceParams* new_params = malloc(sizeof(ReduceParams));
    if (!new_params) return NULL;
    new_params->keepdim = params ? params->keepdim : false;
    new_params->num_dims = params ? params->num_dims : 0;
    if (params && params->dims && params->num_dims > 0) {
        new_params->dims = malloc((size_t)params->num_dims * sizeof(int));
        if (!new_params->dims) { free(new_params); return NULL; }
        memcpy(new_params->dims, params->dims, (size_t)params->num_dims * sizeof(int));
    } else {
        new_params->dims = NULL;
    }

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_ANY, inputs, 1, new_params) != 0) {
        if (new_params->dims) free(new_params->dims);
        free(new_params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int dim = params && params->dims && params->num_dims > 0 ? params->dims[0] : -1;
    bool keepdim = params ? params->keepdim : false;
    int* out_shape = NULL;
    int out_ndim = 0;

    if (dim < 0 || dim >= a->ndim) {
        out_ndim = 1;
        out_shape = malloc(sizeof(int));
        if (!out_shape) return NULL;
        out_shape[0] = 1;
    } else {
        out_ndim = keepdim ? a->ndim : (a->ndim - 1);
        if (out_ndim == 0) out_ndim = 1;
        out_shape = malloc((size_t)out_ndim * sizeof(int));
        if (!out_shape) return NULL;
        int out_idx = 0;
        if (keepdim) {
            for (int i = 0; i < a->ndim; i++)
                out_shape[i] = (i == dim) ? 1 : a->shape[i];
        } else {
            for (int i = 0; i < a->ndim; i++)
                if (i != dim) out_shape[out_idx++] = a->shape[i];
        }
    }

    node->output_shape = out_shape;
    node->output_ndim = out_ndim;
    node->requires_grad = false;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_all(Tensor* a, ReduceParams* params) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    ReduceParams* new_params = malloc(sizeof(ReduceParams));
    if (!new_params) return NULL;
    new_params->keepdim = params ? params->keepdim : false;
    new_params->num_dims = params ? params->num_dims : 0;
    if (params && params->dims && params->num_dims > 0) {
        new_params->dims = malloc((size_t)params->num_dims * sizeof(int));
        if (!new_params->dims) { free(new_params); return NULL; }
        memcpy(new_params->dims, params->dims, (size_t)params->num_dims * sizeof(int));
    } else {
        new_params->dims = NULL;
    }

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_ALL, inputs, 1, new_params) != 0) {
        if (new_params->dims) free(new_params->dims);
        free(new_params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int dim = params && params->dims && params->num_dims > 0 ? params->dims[0] : -1;
    bool keepdim = params ? params->keepdim : false;
    int* out_shape = NULL;
    int out_ndim = 0;

    if (dim < 0 || dim >= a->ndim) {
        out_ndim = 1;
        out_shape = malloc(sizeof(int));
        if (!out_shape) return NULL;
        out_shape[0] = 1;
    } else {
        out_ndim = keepdim ? a->ndim : (a->ndim - 1);
        if (out_ndim == 0) out_ndim = 1;
        out_shape = malloc((size_t)out_ndim * sizeof(int));
        if (!out_shape) return NULL;
        int out_idx = 0;
        if (keepdim) {
            for (int i = 0; i < a->ndim; i++)
                out_shape[i] = (i == dim) ? 1 : a->shape[i];
        } else {
            for (int i = 0; i < a->ndim; i++)
                if (i != dim) out_shape[out_idx++] = a->shape[i];
        }
    }

    node->output_shape = out_shape;
    node->output_ndim = out_ndim;
    node->requires_grad = false;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_logsumexp(Tensor* a, ReduceParams* params) {
    return uop_reduce(a, params, UOP_LOGSUMEXP);
}

Tensor* uop_cummax(Tensor* a, int dim) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    CumsumParams* params = malloc(sizeof(CumsumParams));
    if (!params) return NULL;
    params->dim = dim < 0 ? a->ndim + dim : dim;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_CUMMAX, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_cummin(Tensor* a, int dim) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    CumsumParams* params = malloc(sizeof(CumsumParams));
    if (!params) return NULL;
    params->dim = dim < 0 ? a->ndim + dim : dim;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_CUMMIN, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_cat(Tensor** tensors, int num_tensors, int dim) {
    if (!tensors || num_tensors <= 0) return NULL;
    if (dim < 0) dim = tensors[0]->ndim + dim;
    if (dim < 0 || dim >= tensors[0]->ndim) return NULL;

    int ndim = tensors[0]->ndim;
    for (int t = 1; t < num_tensors; t++) {
        if (!tensors[t] || tensors[t]->ndim != ndim) return NULL;
        for (int d = 0; d < ndim; d++) {
            if (d != dim && tensors[t]->shape[d] != tensors[0]->shape[d]) return NULL;
        }
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    CatParams* params = malloc(sizeof(CatParams));
    if (!params) return NULL;
    params->dim = dim;
    params->num_tensors = num_tensors;

    if (cml_ir_add_uop(ir, UOP_CAT, tensors, num_tensors, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int* out_shape = tensor_shape_copy(tensors[0]->shape, ndim);
    if (!out_shape) return NULL;
    out_shape[dim] = 0;
    for (int t = 0; t < num_tensors; t++)
        out_shape[dim] += tensors[t]->shape[dim];
    node->output_shape = out_shape;
    node->output_ndim = ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_stack(Tensor** tensors, int num_tensors, int dim) {
    if (!tensors || num_tensors <= 0) return NULL;
    int ndim = tensors[0]->ndim;
    if (dim < 0) dim = ndim + 1 + dim;
    if (dim < 0 || dim > ndim) return NULL;

    for (int t = 1; t < num_tensors; t++) {
        if (!tensors[t] || tensors[t]->ndim != ndim) return NULL;
        for (int d = 0; d < ndim; d++) {
            if (tensors[t]->shape[d] != tensors[0]->shape[d]) return NULL;
        }
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    StackParams* params = malloc(sizeof(StackParams));
    if (!params) return NULL;
    params->dim = dim;
    params->num_tensors = num_tensors;

    if (cml_ir_add_uop(ir, UOP_STACK, tensors, num_tensors, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int out_ndim = ndim + 1;
    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape) return NULL;
    int si = 0;
    for (int d = 0; d < out_ndim; d++) {
        if (d == dim) out_shape[d] = num_tensors;
        else out_shape[d] = tensors[0]->shape[si++];
    }
    node->output_shape = out_shape;
    node->output_ndim = out_ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_scatter(Tensor* a, int dim, Tensor* index, Tensor* src) {
    if (!a || !index || !src) return NULL;
    if (dim < 0) dim = a->ndim + dim;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    ScatterParams* params = malloc(sizeof(ScatterParams));
    if (!params) return NULL;
    params->dim = dim;

    Tensor* inputs[] = {a, index, src};
    if (cml_ir_add_uop(ir, UOP_SCATTER, inputs, 3, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_roll(Tensor* a, int shift, int dim) {
    if (!a) return NULL;
    if (dim < 0) dim = a->ndim + dim;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    RollParams* params = malloc(sizeof(RollParams));
    if (!params) return NULL;
    params->shift = shift;
    params->dim = dim;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_ROLL, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_flatten(Tensor* a, int start_dim, int end_dim) {
    if (!a) return NULL;
    if (start_dim < 0) start_dim = a->ndim + start_dim;
    if (end_dim < 0) end_dim = a->ndim + end_dim;
    if (start_dim < 0 || end_dim >= a->ndim || start_dim > end_dim) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    FlattenParams* params = malloc(sizeof(FlattenParams));
    if (!params) return NULL;
    params->start_dim = start_dim;
    params->end_dim = end_dim;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_FLATTEN, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int out_ndim = a->ndim - (end_dim - start_dim);
    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape) return NULL;
    int oi = 0;
    for (int i = 0; i < a->ndim; i++) {
        if (i == start_dim) {
            int flat_size = 1;
            for (int j = start_dim; j <= end_dim; j++) flat_size *= a->shape[j];
            out_shape[oi++] = flat_size;
            i = end_dim;
        } else {
            out_shape[oi++] = a->shape[i];
        }
    }
    node->output_shape = out_shape;
    node->output_ndim = out_ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_unflatten(Tensor* a, int dim, int* sizes, int num_sizes) {
    if (!a || !sizes || num_sizes <= 0) return NULL;
    if (dim < 0) dim = a->ndim + dim;
    if (dim < 0 || dim >= a->ndim) return NULL;

    int product = 1;
    for (int i = 0; i < num_sizes; i++) product *= sizes[i];
    if (product != a->shape[dim]) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    UnflattenParams* params = malloc(sizeof(UnflattenParams));
    if (!params) return NULL;
    params->dim = dim;
    params->num_sizes = num_sizes;
    params->sizes = malloc((size_t)num_sizes * sizeof(int));
    if (!params->sizes) { free(params); return NULL; }
    memcpy(params->sizes, sizes, (size_t)num_sizes * sizeof(int));

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_UNFLATTEN, inputs, 1, params) != 0) {
        free(params->sizes); free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int out_ndim = a->ndim - 1 + num_sizes;
    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape) return NULL;
    int oi = 0;
    for (int i = 0; i < a->ndim; i++) {
        if (i == dim) {
            for (int j = 0; j < num_sizes; j++) out_shape[oi++] = sizes[j];
        } else {
            out_shape[oi++] = a->shape[i];
        }
    }
    node->output_shape = out_shape;
    node->output_ndim = out_ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_diag(Tensor* a, int offset) {
    if (!a) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    DiagParams* params = malloc(sizeof(DiagParams));
    if (!params) return NULL;
    params->offset = offset;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_DIAG, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    if (a->ndim == 1) {
        // 1D -> 2D diagonal matrix
        int n = a->shape[0] + abs(offset);
        int out_shape[] = {n, n};
        node->output_shape = tensor_shape_copy(out_shape, 2);
        node->output_ndim = 2;
    } else if (a->ndim == 2) {
        // 2D -> 1D extract diagonal
        int rows = a->shape[0], cols = a->shape[1];
        int diag_len = 0;
        if (offset >= 0) diag_len = (rows < cols - offset) ? rows : cols - offset;
        else diag_len = (rows + offset < cols) ? rows + offset : cols;
        if (diag_len < 0) diag_len = 0;
        int out_shape[] = {diag_len};
        node->output_shape = tensor_shape_copy(out_shape, 1);
        node->output_ndim = 1;
    } else {
        free(params); return NULL;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_one_hot(Tensor* a, int num_classes) {
    if (!a || num_classes <= 0) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    OneHotParams* params = malloc(sizeof(OneHotParams));
    if (!params) return NULL;
    params->num_classes = num_classes;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_ONE_HOT, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int out_ndim = a->ndim + 1;
    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape) return NULL;
    for (int i = 0; i < a->ndim; i++) out_shape[i] = a->shape[i];
    out_shape[a->ndim] = num_classes;
    node->output_shape = out_shape;
    node->output_ndim = out_ndim;
    node->requires_grad = false;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_erfc(Tensor* a) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;
    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_ERFC, inputs, 1, NULL) != 0) return NULL;
    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_lerp(Tensor* a, Tensor* b, Tensor* t) {
    if (!a || !b || !t) return NULL;
    // lerp(a, b, t) = a + t * (b - a)
    Tensor* diff = uop_sub(b, a);
    if (!diff) return NULL;
    Tensor* scaled = uop_mul(t, diff);
    if (!scaled) return NULL;
    return uop_add(a, scaled);
}

Tensor* uop_tile(Tensor* a, int* repeats, int num_dims) {
    if (!a || !repeats || num_dims != a->ndim) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    TileParams* params = malloc(sizeof(TileParams));
    if (!params) return NULL;
    params->num_dims = num_dims;
    params->repeats = malloc((size_t)num_dims * sizeof(int));
    if (!params->repeats) { free(params); return NULL; }
    memcpy(params->repeats, repeats, (size_t)num_dims * sizeof(int));

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_TILE, inputs, 1, params) != 0) {
        free(params->repeats); free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int* out_shape = malloc((size_t)num_dims * sizeof(int));
    if (!out_shape) return NULL;
    for (int i = 0; i < num_dims; i++)
        out_shape[i] = a->shape[i] * repeats[i];
    node->output_shape = out_shape;
    node->output_ndim = num_dims;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_repeat_interleave(Tensor* a, int repeats, int dim) {
    if (!a || repeats <= 0) return NULL;
    if (dim < 0) dim = a->ndim + dim;
    if (dim < 0 || dim >= a->ndim) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    RepeatInterleaveParams* params = malloc(sizeof(RepeatInterleaveParams));
    if (!params) return NULL;
    params->repeats = repeats;
    params->dim = dim;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_REPEAT_INTERLEAVE, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int* out_shape = tensor_shape_copy(a->shape, a->ndim);
    if (!out_shape) return NULL;
    out_shape[dim] *= repeats;
    node->output_shape = out_shape;
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_trace(Tensor* a) {
    if (!a || a->ndim != 2) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_TRACE, inputs, 1, NULL) != 0) return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    int out_shape[] = {1};
    node->output_shape = tensor_shape_copy(out_shape, 1);
    node->output_ndim = 1;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_shrink(Tensor* a, int* starts, int* ends, int num_dims) {
    if (!a || !starts || !ends || num_dims != a->ndim) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    ShrinkParams* params = malloc(sizeof(ShrinkParams));
    if (!params) return NULL;
    params->num_dims = num_dims;
    params->starts = malloc((size_t)num_dims * sizeof(int));
    params->ends = malloc((size_t)num_dims * sizeof(int));
    if (!params->starts || !params->ends) {
        if (params->starts) free(params->starts);
        if (params->ends) free(params->ends);
        free(params);
        return NULL;
    }
    memcpy(params->starts, starts, (size_t)num_dims * sizeof(int));
    memcpy(params->ends, ends, (size_t)num_dims * sizeof(int));

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_SHRINK, inputs, 1, params) != 0) {
        free(params->starts); free(params->ends); free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int* out_shape = malloc((size_t)num_dims * sizeof(int));
    if (!out_shape) return NULL;
    for (int i = 0; i < num_dims; i++)
        out_shape[i] = ends[i] - starts[i];
    node->output_shape = out_shape;
    node->output_ndim = num_dims;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_logcumsumexp(Tensor* a, int dim) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    CumsumParams* params = malloc(sizeof(CumsumParams));
    if (!params) return NULL;
    params->dim = dim < 0 ? a->ndim + dim : dim;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_LOGCUMSUMEXP, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(a->shape, a->ndim);
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_relu6(Tensor* x) {
    if (!x) return NULL;
    // relu6(x) = min(max(x, 0), 6) = clamp(x, 0, 6)
    return uop_clamp(x, 0.0f, 6.0f);
}

Tensor* uop_hard_sigmoid(Tensor* x) {
    if (!x) return NULL;
    // hard_sigmoid(x) = clamp((x + 3) / 6, 0, 1)
    Tensor* three = uop_fill(x->shape, x->ndim, 3.0f);
    if (!three) return NULL;
    Tensor* shifted = uop_add(x, three);
    if (!shifted) return NULL;
    Tensor* six = uop_fill(x->shape, x->ndim, 6.0f);
    if (!six) return NULL;
    Tensor* scaled = uop_div(shifted, six);
    if (!scaled) return NULL;
    return uop_clamp(scaled, 0.0f, 1.0f);
}

Tensor* uop_hard_tanh(Tensor* x) {
    if (!x) return NULL;
    return uop_clamp(x, -1.0f, 1.0f);
}

Tensor* uop_celu(Tensor* x, float alpha) {
    if (!x) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    ClampParams* params = malloc(sizeof(ClampParams));
    if (!params) return NULL;
    params->min_val = alpha; // store alpha in min_val
    params->max_val = 0.0f;  // unused

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_CELU, inputs, 1, params) != 0) {
        free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(x->shape, x->ndim);
    node->output_ndim = x->ndim;
    if (x->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_quick_gelu(Tensor* x) {
    if (!x) return NULL;
    // quick_gelu(x) = x * sigmoid(1.702 * x)
    Tensor* coeff = uop_fill(x->shape, x->ndim, 1.702f);
    if (!coeff) return NULL;
    Tensor* scaled = uop_mul(coeff, x);
    if (!scaled) return NULL;
    Tensor* sig = uop_sigmoid(scaled);
    if (!sig) return NULL;
    return uop_mul(x, sig);
}

Tensor* uop_softplus(Tensor* x) {
    if (!x) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_SOFTPLUS, inputs, 1, NULL) != 0) return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(x->shape, x->ndim);
    node->output_ndim = x->ndim;
    if (x->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_softsign(Tensor* x) {
    if (!x) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_SOFTSIGN, inputs, 1, NULL) != 0) return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(x->shape, x->ndim);
    node->output_ndim = x->ndim;
    if (x->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_logsigmoid(Tensor* x) {
    if (!x) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_LOGSIGMOID, inputs, 1, NULL) != 0) return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(x->shape, x->ndim);
    node->output_ndim = x->ndim;
    if (x->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_unfold(Tensor* a, int kernel_size, int stride) {
    if (!a) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    UnfoldParams* params = malloc(sizeof(UnfoldParams));
    if (!params) return NULL;
    params->kernel_size = kernel_size;
    params->stride = stride > 0 ? stride : 1;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_UNFOLD, inputs, 1, params) != 0) {
        free(params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    // Output shape: for 1D [L] -> [num_windows, kernel_size]
    // For 2D [N, L] -> [N, num_windows, kernel_size]
    int last_dim = a->shape[a->ndim - 1];
    int num_windows = (last_dim - kernel_size) / params->stride + 1;
    if (num_windows <= 0) {
        LOG_ERROR("uop_unfold: invalid params (kernel_size=%d, stride=%d for dim=%d)",
                  kernel_size, params->stride, last_dim);
        return NULL;
    }

    int out_ndim = a->ndim + 1;
    node->output_shape = malloc((size_t)out_ndim * sizeof(int));
    for (int i = 0; i < a->ndim - 1; i++) {
        node->output_shape[i] = a->shape[i];
    }
    node->output_shape[out_ndim - 2] = num_windows;
    node->output_shape[out_ndim - 1] = kernel_size;
    node->output_ndim = out_ndim;

    return tensor_from_ir_node(node, ir);
}

void uop_var_mean(Tensor* a, ReduceParams* params, Tensor** out_var, Tensor** out_mean) {
    if (!a || !out_var || !out_mean) return;
    *out_mean = uop_mean(a, params);
    *out_var = uop_var(a, params);
}

void uop_std_mean(Tensor* a, ReduceParams* params, Tensor** out_std, Tensor** out_mean) {
    if (!a || !out_std || !out_mean) return;
    *out_mean = uop_mean(a, params);
    *out_std = uop_std(a, params);
}


Tensor* uop_elu(Tensor* x, float alpha) {
    if (!x) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    ClampParams* params = malloc(sizeof(ClampParams));
    if (!params) return NULL;
    params->min_val = alpha;
    params->max_val = 0.0f; // unused

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_ELU, inputs, 1, params) != 0) {
        free(params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(x->shape, x->ndim);
    node->output_ndim = x->ndim;
    if (x->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_selu(Tensor* x) {
    if (!x) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_SELU, inputs, 1, NULL) != 0) return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(x->shape, x->ndim);
    node->output_ndim = x->ndim;
    if (x->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_mish(Tensor* x) {
    if (!x) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_MISH, inputs, 1, NULL) != 0) return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(x->shape, x->ndim);
    node->output_ndim = x->ndim;
    if (x->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_silu(Tensor* x) {
    if (!x) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_SILU, inputs, 1, NULL) != 0) return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(x->shape, x->ndim);
    node->output_ndim = x->ndim;
    if (x->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_hardswish(Tensor* x) {
    if (!x) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {x};
    if (cml_ir_add_uop(ir, UOP_HARDSWISH, inputs, 1, NULL) != 0) return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    node->output_shape = tensor_shape_copy(x->shape, x->ndim);
    node->output_ndim = x->ndim;
    if (x->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}


Tensor* uop_masked_select(Tensor* a, Tensor* mask) {
    if (!a || !mask) return NULL;
    // masked_select flattens and selects where mask is true
    // We need to count nonzeros at execution time, so we use a special UOp
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    Tensor* inputs[] = {a, mask};
    if (cml_ir_add_uop(ir, UOP_MASKED_SELECT, inputs, 2, NULL) != 0) return NULL;

    struct IRNode* node = cml_ir_get_tail(ir);
    // Output is 1D with at most a->numel elements (actual size determined at execution)
    node->output_shape = malloc(sizeof(int));
    if (!node->output_shape) return NULL;
    node->output_shape[0] = (int)a->numel; // upper bound, actual may be less
    node->output_ndim = 1;
    return tensor_from_ir_node(node, ir);
}

Tensor** uop_split(Tensor* a, int split_size, int dim, int* num_splits) {
    if (!a || split_size <= 0 || !num_splits) return NULL;
    if (dim < 0) dim += a->ndim;
    if (dim < 0 || dim >= a->ndim) return NULL;

    int dim_size = a->shape[dim];
    int n_splits = (dim_size + split_size - 1) / split_size;
    *num_splits = n_splits;

    Tensor** results = malloc((size_t)n_splits * sizeof(Tensor*));
    if (!results) return NULL;

    for (int i = 0; i < n_splits; i++) {
        int start = i * split_size;
        int end = start + split_size;
        if (end > dim_size) end = dim_size;

        // Use slice to extract each split
        int* starts = calloc((size_t)a->ndim, sizeof(int));
        int* ends = malloc((size_t)a->ndim * sizeof(int));
        int* steps = malloc((size_t)a->ndim * sizeof(int));
        if (!starts || !ends || !steps) {
            free(starts); free(ends); free(steps);
            free(results);
            return NULL;
        }
        for (int d = 0; d < a->ndim; d++) {
            starts[d] = (d == dim) ? start : 0;
            ends[d] = (d == dim) ? end : a->shape[d];
            steps[d] = 1;
        }

        SliceParams* sp = malloc(sizeof(SliceParams));
        if (!sp) { free(starts); free(ends); free(steps); free(results); return NULL; }
        sp->start = starts;
        sp->end = ends;
        sp->step = steps;
        sp->num_dims = a->ndim;

        results[i] = uop_slice(a, sp);
    }

    return results;
}

Tensor** uop_chunk(Tensor* a, int chunks, int dim, int* num_chunks) {
    if (!a || chunks <= 0 || !num_chunks) return NULL;
    if (dim < 0) dim += a->ndim;
    if (dim < 0 || dim >= a->ndim) return NULL;

    int dim_size = a->shape[dim];
    int chunk_size = (dim_size + chunks - 1) / chunks;
    return uop_split(a, chunk_size, dim, num_chunks);
}

Tensor** uop_meshgrid(Tensor** tensors, int num_tensors, int* num_outputs) {
    if (!tensors || num_tensors <= 0 || !num_outputs) return NULL;
    *num_outputs = num_tensors;

    // Compute output shape: [len(t0), len(t1), ...]
    int* out_shape = malloc((size_t)num_tensors * sizeof(int));
    if (!out_shape) return NULL;
    for (int i = 0; i < num_tensors; i++) {
        if (!tensors[i]) { free(out_shape); return NULL; }
        out_shape[i] = (int)tensors[i]->numel;
    }

    Tensor** results = malloc((size_t)num_tensors * sizeof(Tensor*));
    if (!results) { free(out_shape); return NULL; }

    for (int i = 0; i < num_tensors; i++) {
        // Reshape tensor[i] so it broadcasts to out_shape
        int* reshape = malloc((size_t)num_tensors * sizeof(int));
        if (!reshape) { free(out_shape); free(results); return NULL; }
        for (int j = 0; j < num_tensors; j++) {
            reshape[j] = (i == j) ? out_shape[j] : 1;
        }
        Tensor* reshaped = uop_reshape(tensors[i], &(ReshapeParams){.new_shape = reshape, .new_ndim = num_tensors});
        free(reshape);
        if (!reshaped) { free(out_shape); free(results); return NULL; }

        // Expand to full shape
        ExpandParams ep = {.new_shape = out_shape, .new_ndim = num_tensors};
        results[i] = uop_expand(reshaped, &ep);
    }

    free(out_shape);
    return results;
}

Tensor* uop_diagonal(Tensor* a, int offset, int dim1, int dim2) {
    if (!a || a->ndim < 2) return NULL;
    if (dim1 < 0) dim1 += a->ndim;
    if (dim2 < 0) dim2 += a->ndim;
    if (dim1 < 0 || dim1 >= a->ndim || dim2 < 0 || dim2 >= a->ndim || dim1 == dim2) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    DiagParams* params = malloc(sizeof(DiagParams));
    if (!params) return NULL;
    params->offset = offset;

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_DIAGONAL, inputs, 1, params) != 0) {
        free(params);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);

    // Compute output shape: replace dim1,dim2 with diag_size
    int rows = a->shape[dim1];
    int cols = a->shape[dim2];
    int diag_size;
    if (offset >= 0) {
        diag_size = rows < (cols - offset) ? rows : (cols - offset);
    } else {
        diag_size = (rows + offset) < cols ? (rows + offset) : cols;
    }
    if (diag_size < 0) diag_size = 0;

    // Output has ndim-1 dimensions
    int out_ndim = a->ndim - 1;
    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape) return NULL;
    int oi = 0;
    for (int d = 0; d < a->ndim; d++) {
        if (d == dim1) {
            out_shape[oi++] = diag_size;
        } else if (d == dim2) {
            continue; // skip
        } else {
            out_shape[oi++] = a->shape[d];
        }
    }
    node->output_shape = out_shape;
    node->output_ndim = out_ndim;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_pad_reflect(Tensor* a, int* pad_widths, int num_dims) {
    if (!a || !pad_widths || num_dims != a->ndim) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    PadParams* params = malloc(sizeof(PadParams));
    if (!params) return NULL;
    params->num_dims = num_dims;
    params->value = 0.0f;
    params->mode = PAD_REFLECT;
    params->pad_widths = malloc((size_t)(num_dims * 2) * sizeof(int));
    if (!params->pad_widths) { free(params); return NULL; }
    memcpy(params->pad_widths, pad_widths, (size_t)(num_dims * 2) * sizeof(int));

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_PAD, inputs, 1, params) != 0) {
        free(params->pad_widths); free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int* out_shape = malloc((size_t)a->ndim * sizeof(int));
    if (!out_shape) return NULL;
    for (int i = 0; i < a->ndim; i++) {
        out_shape[i] = a->shape[i] + pad_widths[i * 2] + pad_widths[i * 2 + 1];
    }
    node->output_shape = out_shape;
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_pad_replicate(Tensor* a, int* pad_widths, int num_dims) {
    if (!a || !pad_widths || num_dims != a->ndim) return NULL;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    PadParams* params = malloc(sizeof(PadParams));
    if (!params) return NULL;
    params->num_dims = num_dims;
    params->value = 0.0f;
    params->mode = PAD_REPLICATE;
    params->pad_widths = malloc((size_t)(num_dims * 2) * sizeof(int));
    if (!params->pad_widths) { free(params); return NULL; }
    memcpy(params->pad_widths, pad_widths, (size_t)(num_dims * 2) * sizeof(int));

    Tensor* inputs[] = {a};
    if (cml_ir_add_uop(ir, UOP_PAD, inputs, 1, params) != 0) {
        free(params->pad_widths); free(params); return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int* out_shape = malloc((size_t)a->ndim * sizeof(int));
    if (!out_shape) return NULL;
    for (int i = 0; i < a->ndim; i++) {
        out_shape[i] = a->shape[i] + pad_widths[i * 2] + pad_widths[i * 2 + 1];
    }
    node->output_shape = out_shape;
    node->output_ndim = a->ndim;
    if (a->requires_grad) {
        node->requires_grad = true;
        node->needs_input_grad[0] = true;
    }
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_scaled_dot_product_attention(Tensor* q, Tensor* k, Tensor* v, Tensor* mask) {
    if (!q || !k || !v) return NULL;
    // SDPA: softmax(Q*K^T / sqrt(d_k)) * V
    // Q: [..., seq_len, d_k], K: [..., seq_len, d_k], V: [..., seq_len, d_v]

    int d_k = q->shape[q->ndim - 1];
    float scale = 1.0f / sqrtf((float)d_k);

    // K^T: permute last two dims
    PermuteParams perm_params = {0};
    int* perm = malloc((size_t)k->ndim * sizeof(int));
    if (!perm) return NULL;
    for (int i = 0; i < k->ndim - 2; i++) perm[i] = i;
    perm[k->ndim - 2] = k->ndim - 1;
    perm[k->ndim - 1] = k->ndim - 2;
    perm_params.perm = perm;
    perm_params.num_dims = k->ndim;

    Tensor* kt = uop_permute(k, &perm_params);
    free(perm);
    if (!kt) return NULL;

    // Q * K^T
    Tensor* scores = uop_matmul(q, kt);
    if (!scores) return NULL;

    // / sqrt(d_k)
    Tensor* scale_t = uop_fill(scores->shape, scores->ndim, scale);
    if (!scale_t) return NULL;
    Tensor* scaled = uop_mul(scores, scale_t);
    if (!scaled) return NULL;

    // Apply mask if provided
    if (mask) {
        scaled = uop_masked_fill(scaled, mask, -1e9f);
        if (!scaled) return NULL;
    }

    // softmax along last dim
    Tensor* attn = uop_softmax(scaled, scaled->ndim - 1);
    if (!attn) return NULL;

    // * V
    return uop_matmul(attn, v);
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
    case UOP_SIGN:
        if (num_inputs >= 1) return uop_sign(inputs[0]);
        break;
    case UOP_FLOOR:
        if (num_inputs >= 1) return uop_floor(inputs[0]);
        break;
    case UOP_CEIL:
        if (num_inputs >= 1) return uop_ceil(inputs[0]);
        break;
    case UOP_ROUND:
        if (num_inputs >= 1) return uop_round(inputs[0]);
        break;
    case UOP_LOG2:
        if (num_inputs >= 1) return uop_log2(inputs[0]);
        break;
    case UOP_EXP2:
        if (num_inputs >= 1) return uop_exp2(inputs[0]);
        break;
    case UOP_ASIN:
        if (num_inputs >= 1) return uop_asin(inputs[0]);
        break;
    case UOP_ACOS:
        if (num_inputs >= 1) return uop_acos(inputs[0]);
        break;
    case UOP_ATAN:
        if (num_inputs >= 1) return uop_atan(inputs[0]);
        break;
    case UOP_SQUARE:
        if (num_inputs >= 1) return uop_square(inputs[0]);
        break;
    case UOP_RSQRT:
        if (num_inputs >= 1) return uop_rsqrt(inputs[0]);
        break;
    case UOP_ERF:
        if (num_inputs >= 1) return uop_erf(inputs[0]);
        break;
    case UOP_CLAMP:
        if (num_inputs >= 1 && params) {
            ClampParams* cp = (ClampParams*)params;
            return uop_clamp(inputs[0], cp->min_val, cp->max_val);
        }
        break;
    case UOP_PROD:
        if (num_inputs >= 1) return uop_prod(inputs[0], (ReduceParams*)params);
        break;
    case UOP_ARGMAX:
        if (num_inputs >= 1) return uop_argmax(inputs[0], (ReduceParams*)params);
        break;
    case UOP_ARGMIN:
        if (num_inputs >= 1) return uop_argmin(inputs[0], (ReduceParams*)params);
        break;
    case UOP_CUMSUM:
        if (num_inputs >= 1 && params) {
            CumsumParams* cp = (CumsumParams*)params;
            return uop_cumsum(inputs[0], cp->dim);
        }
        break;
    case UOP_TRIU:
        if (num_inputs >= 1 && params) {
            TriParams* tp = (TriParams*)params;
            return uop_triu(inputs[0], tp->diagonal);
        }
        break;
    case UOP_TRIL:
        if (num_inputs >= 1 && params) {
            TriParams* tp = (TriParams*)params;
            return uop_tril(inputs[0], tp->diagonal);
        }
        break;
    case UOP_PAD:
        if (num_inputs >= 1 && params) {
            PadParams* pp = (PadParams*)params;
            return uop_pad(inputs[0], pp->pad_widths, pp->num_dims, pp->value);
        }
        break;
    case UOP_FILL:
        // UOP_FILL is a source node, handled separately
        break;
    case UOP_GATHER:
        if (num_inputs >= 2 && params) {
            GatherParams* gp = (GatherParams*)params;
            return uop_gather(inputs[0], inputs[1], gp->dim);
        }
        break;
    case UOP_SORT:
        if (num_inputs >= 1 && params) {
            SortParams* sp = (SortParams*)params;
            return uop_sort(inputs[0], sp->dim, sp->descending);
        }
        break;
    case UOP_ARGSORT:
        if (num_inputs >= 1 && params) {
            SortParams* sp = (SortParams*)params;
            return uop_argsort(inputs[0], sp->dim, sp->descending);
        }
        break;
    case UOP_TOPK:
        if (num_inputs >= 1 && params) {
            TopkParams* tp = (TopkParams*)params;
            return uop_topk(inputs[0], tp->k, tp->dim, tp->largest, NULL);
        }
        break;
    case UOP_CUMPROD:
        if (num_inputs >= 1 && params) {
            CumsumParams* cp = (CumsumParams*)params;
            return uop_cumprod(inputs[0], cp->dim);
        }
        break;
    case UOP_BITWISE_AND:
        if (num_inputs >= 2) return uop_bitwise_and(inputs[0], inputs[1]);
        break;
    case UOP_BITWISE_OR:
        if (num_inputs >= 2) return uop_bitwise_or(inputs[0], inputs[1]);
        break;
    case UOP_BITWISE_XOR:
        if (num_inputs >= 2) return uop_bitwise_xor(inputs[0], inputs[1]);
        break;
    case UOP_BITWISE_NOT:
        if (num_inputs >= 1) return uop_bitwise_not(inputs[0]);
        break;
    case UOP_NONZERO:
        if (num_inputs >= 1) return uop_nonzero(inputs[0]);
        break;
    case UOP_MASKED_FILL:
        if (num_inputs >= 2 && params) {
            MaskedFillParams* mfp = (MaskedFillParams*)params;
            return uop_masked_fill(inputs[0], inputs[1], mfp->value);
        }
        break;
    case UOP_UNFOLD:
        if (num_inputs >= 1 && params) {
            UnfoldParams* up = (UnfoldParams*)params;
            return uop_unfold(inputs[0], up->kernel_size, up->stride);
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
