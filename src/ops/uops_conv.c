/*
 * Convolution and pooling uop builders (conv2d/3d, transposed conv, max/avg
 * pool). Split out of uops.c; the cluster is self-contained (its only shared
 * helper, conv2d_params_free_local, moved with it) and finalizes nodes via the
 * public IR API directly.
 */
#include "ops/uops.h"
#include "ops/winograd.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include "core/error_stack.h"
#include "core/error_codes.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alloc/cml_allocator.h"

static void conv2d_params_free_local(Conv2DParams* p) {
    if (!p)
        return;
    if (p->kernel_size)
        cml_free(p->kernel_size);
    if (p->stride)
        cml_free(p->stride);
    if (p->padding)
        cml_free(p->padding);
    if (p->dilation)
        cml_free(p->dilation);
    cml_free(p);
}

static Tensor* uop_pool2d(Tensor* input, Pool2DParams* params, UOpType type) {
    if (!input || !params) {
        CML_ERR_NULL("NULL input to uop_pool2d");
    }
    if (input->ndim != 4) {
        LOG_ERROR("Pool2D expects 4D input [batch, channels, height, width], got %dD", input->ndim);
        error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
        return NULL;
    }

    int kernel_h   = params->kernel_size[0];
    int kernel_w   = params->kernel_size[1];
    int stride_h   = params->stride[0] > 0 ? params->stride[0] : kernel_h;
    int stride_w   = params->stride[1] > 0 ? params->stride[1] : kernel_w;
    int padding_h  = params->padding[0];
    int padding_w  = params->padding[1];
    int dilation_h = params->dilation[0] > 0 ? params->dilation[0] : 1;
    int dilation_w = params->dilation[1] > 0 ? params->dilation[1] : 1;
    int in_height  = input->shape[2];
    int in_width   = input->shape[3];

    if (kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 ||
        dilation_w <= 0) {
        CML_ERR_NULL("uop_pool2d: invalid kernel/stride/dilation");
    }

    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width  = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    if (params->ceil_mode) {
        int numer_h = in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1;
        int numer_w = in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1;
        if ((numer_h % stride_h) != 0)
            out_height++;
        if ((numer_w % stride_w) != 0)
            out_width++;
    }
    if (out_height <= 0 || out_width <= 0) {
        LOG_ERROR("uop_pool2d: invalid output dimensions (%d x %d)", out_height, out_width);
        error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
        return NULL;
    }

    Pool2DParams* params_copy = cml_malloc(sizeof(Pool2DParams));
    if (!params_copy)
        return NULL;
    memcpy(params_copy, params, sizeof(Pool2DParams));
    params_copy->stride[0]   = stride_h;
    params_copy->stride[1]   = stride_w;
    params_copy->dilation[0] = dilation_h;
    params_copy->dilation[1] = dilation_w;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        cml_free(params_copy);
        return NULL;
    }

    Tensor* inputs[] = {input};
    if (cml_ir_add_uop(ir, type, inputs, 1, params_copy) != 0) {
        cml_free(params_copy);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    if (!node)
        return NULL;

    int output_shape[4] = {input->shape[0], input->shape[1], out_height, out_width};
    node->output_shape  = tensor_shape_copy(output_shape, 4);
    node->output_ndim   = 4;
    if (input->requires_grad) {
        node->requires_grad       = true;
        node->needs_input_grad[0] = true;
    }

    return tensor_from_ir_node(node, ir);
}

static Tensor* uop_conv3d_like(Tensor* input, Tensor* weight, Tensor* bias,
                               const Conv3DParams* params) {
    if (!input || !weight) {
        CML_ERR_NULL("NULL tensor input to uop_conv3d");
    }
    if (input->ndim != 5 || weight->ndim != 5) {
        CML_ERR_NULL("Conv3D expects input [N,C,D,H,W] and weight [O,C,Kd,Kh,Kw]");
    }

    int batch              = input->shape[0];
    int in_channels        = input->shape[1];
    int in_depth           = input->shape[2];
    int in_height          = input->shape[3];
    int in_width           = input->shape[4];
    int out_channels       = weight->shape[0];
    int weight_in_channels = weight->shape[1];
    int kernel_d           = weight->shape[2];
    int kernel_h           = weight->shape[3];
    int kernel_w           = weight->shape[4];

    if (in_channels != weight_in_channels) {
        LOG_ERROR("Conv3D: input channels (%d) don't match weight channels (%d)", in_channels,
                  weight_in_channels);
        error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
        return NULL;
    }

    int stride_d   = params ? params->stride[0] : 1;
    int stride_h   = params ? params->stride[1] : 1;
    int stride_w   = params ? params->stride[2] : 1;
    int pad_d      = params ? params->padding[0] : 0;
    int pad_h      = params ? params->padding[1] : 0;
    int pad_w      = params ? params->padding[2] : 0;
    int dilation_d = params ? params->dilation[0] : 1;
    int dilation_h = params ? params->dilation[1] : 1;
    int dilation_w = params ? params->dilation[2] : 1;

    int out_depth  = (in_depth + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width  = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    if (out_depth <= 0 || out_height <= 0 || out_width <= 0)
        return NULL;

    Conv3DParams* params_copy = cml_calloc(1, sizeof(Conv3DParams));
    if (!params_copy)
        return NULL;
    if (params)
        memcpy(params_copy, params, sizeof(Conv3DParams));
    params_copy->kernel_size[0] = kernel_d;
    params_copy->kernel_size[1] = kernel_h;
    params_copy->kernel_size[2] = kernel_w;
    if (!params) {
        params_copy->stride[0] = params_copy->stride[1] = params_copy->stride[2] = 1;
        params_copy->dilation[0] = params_copy->dilation[1] = params_copy->dilation[2] = 1;
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        cml_free(params_copy);
        return NULL;
    }

    Tensor* inputs[3] = {input, weight, bias};
    int num_inputs    = bias ? 3 : 2;
    if (cml_ir_add_uop(ir, UOP_CONV3D, inputs, num_inputs, params_copy) != 0) {
        cml_free(params_copy);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int output_shape[5] = {batch, out_channels, out_depth, out_height, out_width};
    node->output_shape  = tensor_shape_copy(output_shape, 5);
    node->output_ndim   = 5;
    return tensor_from_ir_node(node, ir);
}

static Tensor* uop_conv_transpose2d_like(Tensor* input, Tensor* weight, Tensor* bias,
                                         const ConvTranspose2DParams* params) {
    if (!input || !weight || !params) {
        CML_ERR_NULL("NULL input to uop_conv_transpose2d");
    }
    if (input->ndim != 4 || weight->ndim != 4) {
        CML_ERR_NULL("ConvTranspose2D expects input [N,C,H,W] and weight [Cin,Cout,Kh,Kw]");
    }

    int batch              = input->shape[0];
    int in_channels        = input->shape[1];
    int in_height          = input->shape[2];
    int in_width           = input->shape[3];
    int weight_in_channels = weight->shape[0];
    int out_channels       = weight->shape[1];
    int kernel_h           = weight->shape[2];
    int kernel_w           = weight->shape[3];
    if (in_channels != weight_in_channels)
        return NULL;

    int out_height = (in_height - 1) * params->stride[0] - 2 * params->padding[0] +
                     params->dilation[0] * (kernel_h - 1) + params->output_padding[0] + 1;
    int out_width = (in_width - 1) * params->stride[1] - 2 * params->padding[1] +
                    params->dilation[1] * (kernel_w - 1) + params->output_padding[1] + 1;
    if (out_height <= 0 || out_width <= 0)
        return NULL;

    ConvTranspose2DParams* params_copy = cml_malloc(sizeof(ConvTranspose2DParams));
    if (!params_copy)
        return NULL;
    memcpy(params_copy, params, sizeof(ConvTranspose2DParams));
    params_copy->kernel_size[0] = kernel_h;
    params_copy->kernel_size[1] = kernel_w;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        cml_free(params_copy);
        return NULL;
    }
    Tensor* inputs[3] = {input, weight, bias};
    int num_inputs    = bias ? 3 : 2;
    if (cml_ir_add_uop(ir, UOP_CONV_TRANSPOSE2D, inputs, num_inputs, params_copy) != 0) {
        cml_free(params_copy);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int output_shape[4] = {batch, out_channels, out_height, out_width};
    node->output_shape  = tensor_shape_copy(output_shape, 4);
    node->output_ndim   = 4;
    return tensor_from_ir_node(node, ir);
}

static Tensor* uop_conv_transpose3d_like(Tensor* input, Tensor* weight, Tensor* bias,
                                         const ConvTranspose3DParams* params) {
    if (!input || !weight || !params) {
        CML_ERR_NULL("NULL input to uop_conv_transpose3d");
    }
    if (input->ndim != 5 || weight->ndim != 5) {
        CML_ERR_NULL("ConvTranspose3D expects input [N,C,D,H,W] and weight [Cin,Cout,Kd,Kh,Kw]");
    }

    int batch              = input->shape[0];
    int in_channels        = input->shape[1];
    int in_depth           = input->shape[2];
    int in_height          = input->shape[3];
    int in_width           = input->shape[4];
    int weight_in_channels = weight->shape[0];
    int out_channels       = weight->shape[1];
    int kernel_d           = weight->shape[2];
    int kernel_h           = weight->shape[3];
    int kernel_w           = weight->shape[4];
    if (in_channels != weight_in_channels)
        return NULL;

    int out_depth = (in_depth - 1) * params->stride[0] - 2 * params->padding[0] +
                    params->dilation[0] * (kernel_d - 1) + params->output_padding[0] + 1;
    int out_height = (in_height - 1) * params->stride[1] - 2 * params->padding[1] +
                     params->dilation[1] * (kernel_h - 1) + params->output_padding[1] + 1;
    int out_width = (in_width - 1) * params->stride[2] - 2 * params->padding[2] +
                    params->dilation[2] * (kernel_w - 1) + params->output_padding[2] + 1;
    if (out_depth <= 0 || out_height <= 0 || out_width <= 0)
        return NULL;

    ConvTranspose3DParams* params_copy = cml_malloc(sizeof(ConvTranspose3DParams));
    if (!params_copy)
        return NULL;
    memcpy(params_copy, params, sizeof(ConvTranspose3DParams));
    params_copy->kernel_size[0] = kernel_d;
    params_copy->kernel_size[1] = kernel_h;
    params_copy->kernel_size[2] = kernel_w;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        cml_free(params_copy);
        return NULL;
    }
    Tensor* inputs[3] = {input, weight, bias};
    int num_inputs    = bias ? 3 : 2;
    if (cml_ir_add_uop(ir, UOP_CONV_TRANSPOSE3D, inputs, num_inputs, params_copy) != 0) {
        cml_free(params_copy);
        return NULL;
    }

    struct IRNode* node = cml_ir_get_tail(ir);
    int output_shape[5] = {batch, out_channels, out_depth, out_height, out_width};
    node->output_shape  = tensor_shape_copy(output_shape, 5);
    node->output_ndim   = 5;
    return tensor_from_ir_node(node, ir);
}

Tensor* uop_conv2d(Tensor* input, Tensor* weight, Tensor* bias, Conv2DParams* params) {
    if (!input || !weight) {
        LOG_ERROR("NULL tensor input to uop_conv2d");
        error_stack_push(CM_INVALID_ARGUMENT, "uop_conv2d: NULL tensor input", __FILE__, __LINE__,
                         __func__);
        error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (input->ndim != 4) {
        LOG_ERROR("Conv2D expects 4D input [batch, in_channels, height, width], got %dD",
                  input->ndim);
        error_stack_push(CM_INVALID_ARGUMENT, "uop_conv2d: invalid input rank", __FILE__, __LINE__,
                         __func__);
        error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (weight->ndim != 4) {
        LOG_ERROR("Conv2D weight must be 4D [out_channels, in_channels, kernel_h, kernel_w], got "
                  "%dD",
                  weight->ndim);
        error_stack_push(CM_INVALID_ARGUMENT, "uop_conv2d: invalid weight rank", __FILE__, __LINE__,
                         __func__);
        error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
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
        error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
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
        error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (bias) {
        if (bias->ndim != 1 || bias->shape[0] != out_channels) {
            LOG_ERROR("Conv2D bias must be 1D with shape [out_channels]");
            error_stack_push(CM_INVALID_ARGUMENT, "uop_conv2d: invalid bias shape", __FILE__,
                             __LINE__, __func__);
            error_stack_push(CM_INVALID_ARGUMENT, "Operation failed", __FILE__, __LINE__, __func__);
            return NULL;
        }
    }

    Conv2DParams* params_copy = cml_calloc(1, sizeof(Conv2DParams));
    if (!params_copy) {
        error_stack_push(CM_OPERATION_FAILED, "uop_conv2d: params alloc failed", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    params_copy->kernel_size = cml_malloc(2 * sizeof(int));
    params_copy->stride      = cml_malloc(2 * sizeof(int));
    params_copy->padding     = cml_malloc(2 * sizeof(int));
    params_copy->dilation    = cml_malloc(2 * sizeof(int));
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

    params_copy->use_winograd =
        winograd_applicable(kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w);

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

Tensor* uop_maxpool2d(Tensor* input, Pool2DParams* params) {
    return uop_pool2d(input, params, UOP_MAXPOOL2D);
}

Tensor* uop_avgpool2d(Tensor* input, Pool2DParams* params) {
    return uop_pool2d(input, params, UOP_AVGPOOL2D);
}

Tensor* uop_conv3d(Tensor* input, Tensor* weight, Tensor* bias, Conv3DParams* params) {
    return uop_conv3d_like(input, weight, bias, params);
}

Tensor* uop_conv_transpose2d(Tensor* input, Tensor* weight, Tensor* bias,
                             ConvTranspose2DParams* params) {
    return uop_conv_transpose2d_like(input, weight, bias, params);
}

Tensor* uop_conv_transpose3d(Tensor* input, Tensor* weight, Tensor* bias,
                             ConvTranspose3DParams* params) {
    return uop_conv_transpose3d_like(input, weight, bias, params);
}
