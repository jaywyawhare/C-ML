#include "nn/layers/layernorm.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "autograd/autograd.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor* layernorm_forward(Module* module, Tensor* input) {
    LayerNorm* ln = (LayerNorm*)module;

    if (!ln || !input)
        return NULL;

    if (input->ndim < 1) {
        LOG_ERROR("LayerNorm expects at least 1D input, got %dD", input->ndim);
        return NULL;
    }

    int last_dim = input->shape[input->ndim - 1];

    if (last_dim != ln->normalized_shape) {
        LOG_ERROR("LayerNorm: input last dimension (%d) doesn't match normalized_shape (%d)",
                  last_dim, ln->normalized_shape);
        return NULL;
    }
    ReduceParams mean_params;
    int mean_dim         = input->ndim - 1;
    int mean_dims[]      = {mean_dim};
    mean_params.dims     = mean_dims;
    mean_params.num_dims = 1;
    mean_params.keepdim  = true;

    Tensor* mean_reduced = uop_mean(input, &mean_params);
    if (!mean_reduced)
        return NULL;

    Tensor* centered = uop_sub(input, mean_reduced);
    if (!centered)
        return NULL;

    Tensor* diff_sq = uop_mul(centered, centered);
    if (!diff_sq)
        return NULL;

    ReduceParams var_params;
    int var_dims[]      = {mean_dim};
    var_params.dims     = var_dims;
    var_params.num_dims = 1;
    var_params.keepdim  = true;

    Tensor* var_reduced = uop_mean(diff_sq, &var_params);
    if (!var_reduced)
        return NULL;

    Tensor* eps_tensor = uop_fill(var_reduced->shape, var_reduced->ndim, ln->eps);
    if (!eps_tensor)
        return NULL;

    Tensor* var_eps = uop_add(var_reduced, eps_tensor);
    if (!var_eps)
        return NULL;

    Tensor* std_tensor = uop_sqrt(var_eps);
    if (!std_tensor)
        return NULL;

    Tensor* normalized = uop_div(centered, std_tensor);
    if (!normalized)
        return NULL;

    Tensor* output = normalized;

    if (ln->affine && ln->weight && ln->bias) {
        ExpandParams expand_weight;
        expand_weight.new_shape = input->shape;
        expand_weight.new_ndim  = input->ndim;

        Tensor* weight_broadcast = uop_expand(ln->weight->tensor, &expand_weight);
        if (!weight_broadcast)
            return NULL;

        Tensor* scaled = uop_mul(weight_broadcast, output);
        if (!scaled)
            return NULL;

        ExpandParams expand_bias;
        expand_bias.new_shape = input->shape;
        expand_bias.new_ndim  = input->ndim;

        Tensor* bias_broadcast = uop_expand(ln->bias->tensor, &expand_bias);
        if (!bias_broadcast)
            return NULL;

        output = uop_add(scaled, bias_broadcast);
        if (!output)
            return NULL;
    }
    if (autograd_is_grad_enabled() && input->requires_grad) {
        output->requires_grad = true;
    }

    return output;
}

static void layernorm_free(Module* module) {
    LayerNorm* ln = (LayerNorm*)module;
    if (!ln)
        return;

    module_free(module);
}

LayerNorm* nn_layernorm(int normalized_shape, float eps, bool affine, DType dtype,
                        DeviceType device) {
    if (normalized_shape <= 0) {
        LOG_ERROR("LayerNorm: normalized_shape must be positive, got %d", normalized_shape);
        return NULL;
    }

    LayerNorm* ln = malloc(sizeof(LayerNorm));
    if (!ln)
        return NULL;
    if (module_init((Module*)ln, "LayerNorm", layernorm_forward, layernorm_free) != 0) {
        free(ln);
        return NULL;
    }
    ln->normalized_shape = normalized_shape;
    ln->eps              = eps > 0.0f ? eps : 1e-5f;
    ln->affine           = affine;
    ln->weight           = NULL;
    ln->bias             = NULL;
    if (affine) {
        int weight_shape[] = {normalized_shape};

        TensorConfig config =
            (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
        Tensor* weight = tensor_ones(weight_shape, 1, &config);
        if (!weight) {
            module_free((Module*)ln);
            return NULL;
        }
        if (module_add_parameter((Module*)ln, weight, "weight", true) != 0) {
            tensor_free(weight);
            module_free((Module*)ln);
            return NULL;
        }

        ln->weight = module_get_parameter((Module*)ln, "weight");

        Tensor* bias = tensor_zeros(weight_shape, 1, &config);
        if (!bias) {
            module_free((Module*)ln);
            return NULL;
        }
        if (module_add_parameter((Module*)ln, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)ln);
            return NULL;
        }

        ln->bias = module_get_parameter((Module*)ln, "bias");
    } else {
        ln->weight = NULL;
        ln->bias   = NULL;
    }

    return ln;
}
