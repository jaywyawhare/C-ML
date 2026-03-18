#include "nn/layers/rmsnorm.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "autograd/autograd.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor* rmsnorm_forward(Module* module, Tensor* input) {
    RMSNorm* rn = (RMSNorm*)module;

    if (!rn || !input)
        return NULL;

    if (input->ndim < 1) {
        LOG_ERROR("RMSNorm expects at least 1D input, got %dD", input->ndim);
        return NULL;
    }

    int last_dim = input->shape[input->ndim - 1];

    if (last_dim != rn->normalized_shape) {
        LOG_ERROR("RMSNorm: input last dimension (%d) doesn't match normalized_shape (%d)",
                  last_dim, rn->normalized_shape);
        return NULL;
    }
    Tensor* input_sq = uop_mul(input, input);
    if (!input_sq)
        return NULL;
    ReduceParams mean_params;
    int mean_dim         = input->ndim - 1;
    int mean_dims[]      = {mean_dim};
    mean_params.dims     = mean_dims;
    mean_params.num_dims = 1;
    mean_params.keepdim  = true;

    Tensor* mean_sq = uop_mean(input_sq, &mean_params);
    if (!mean_sq)
        return NULL;
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* eps_tensor = tensor_zeros(mean_sq->shape, mean_sq->ndim, &config);
    if (!eps_tensor)
        return NULL;

    float* eps_data = (float*)tensor_data_ptr(eps_tensor);
    if (eps_data) {
        for (size_t i = 0; i < eps_tensor->numel; i++) {
            eps_data[i] = rn->eps;
        }
    }

    Tensor* mean_sq_eps = uop_add(mean_sq, eps_tensor);
    if (!mean_sq_eps)
        return NULL;
    Tensor* rms = uop_sqrt(mean_sq_eps);
    if (!rms)
        return NULL;
    Tensor* normalized = uop_div(input, rms);
    if (!normalized)
        return NULL;
    Tensor* output = normalized;
    if (rn->weight && rn->weight->tensor) {
        ExpandParams expand_weight;
        expand_weight.new_shape = input->shape;
        expand_weight.new_ndim  = input->ndim;

        Tensor* weight_broadcast = uop_expand(rn->weight->tensor, &expand_weight);
        if (!weight_broadcast) {
            tensor_free(output);
            return NULL;
        }

        Tensor* scaled = uop_mul(weight_broadcast, output);
        tensor_free(weight_broadcast);
        tensor_free(output);
        if (!scaled)
            return NULL;

        output = scaled;
    }

    if (autograd_is_grad_enabled() && input->requires_grad) {
        output->requires_grad = true;
    }

    return output;
}

static void rmsnorm_free(Module* module) {
    RMSNorm* rn = (RMSNorm*)module;
    if (!rn)
        return;
    module_free(module);
}

RMSNorm* nn_rmsnorm(int normalized_shape, float eps, DType dtype, DeviceType device) {
    if (normalized_shape <= 0) {
        LOG_ERROR("RMSNorm: normalized_shape must be positive, got %d", normalized_shape);
        return NULL;
    }

    RMSNorm* rn = malloc(sizeof(RMSNorm));
    if (!rn)
        return NULL;

    if (module_init((Module*)rn, "RMSNorm", rmsnorm_forward, rmsnorm_free) != 0) {
        free(rn);
        return NULL;
    }

    rn->normalized_shape = normalized_shape;
    rn->eps              = eps > 0.0f ? eps : 1e-5f;
    rn->weight           = NULL;
    int weight_shape[] = {normalized_shape};
    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* weight = tensor_ones(weight_shape, 1, &config);
    if (!weight) {
        module_free((Module*)rn);
        return NULL;
    }

    if (module_add_parameter((Module*)rn, weight, "weight", true) != 0) {
        tensor_free(weight);
        module_free((Module*)rn);
        return NULL;
    }

    rn->weight = module_get_parameter((Module*)rn, "weight");

    return rn;
}
