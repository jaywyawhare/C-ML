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
#include "alloc/cml_allocator.h"

/** RMSNorm forward: divide by the root-mean-square over the last dim (with eps), then apply the
 *  optional learnable weight (no mean subtraction, no bias). NULL on rank or last-dim mismatch. */
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
    if (!rn->weight || !rn->weight->tensor) {
        LOG_ERROR("RMSNorm: missing weight parameter");
        return NULL;
    }

    /* One fused kernel (x * rsqrt(mean(x^2)+eps) * weight) instead of a
     * materialized mul/mean/add/sqrt/div/mul chain; see uop_rmsnorm. */
    Tensor* output = uop_rmsnorm(input, rn->weight->tensor, rn->eps);
    if (!output)
        return NULL;

    if (autograd_is_grad_enabled() && input->requires_grad) {
        output->requires_grad = true;
    }

    return output;
}

/** Free the RMSNorm module; the weight parameter is released by module_free. */
static void rmsnorm_free(Module* module) {
    RMSNorm* rn = (RMSNorm*)module;
    if (!rn)
        return;
    module_free(module);
}

/** Construct an RMSNorm over a last dim of size normalized_shape; weight is initialized to ones.
 *  eps falls back to 1e-5 if non-positive. NULL on bad shape or failure. */
RMSNorm* nn_rmsnorm(int normalized_shape, float eps, DType dtype, DeviceType device) {
    if (normalized_shape <= 0) {
        LOG_ERROR("RMSNorm: normalized_shape must be positive, got %d", normalized_shape);
        return NULL;
    }

    RMSNorm* rn = cml_malloc(sizeof(RMSNorm));
    if (!rn)
        return NULL;

    if (module_init((Module*)rn, "RMSNorm", rmsnorm_forward, rmsnorm_free) != 0) {
        cml_free(rn);
        return NULL;
    }

    rn->normalized_shape = normalized_shape;
    rn->eps              = eps > 0.0f ? eps : 1e-5f;
    rn->weight           = NULL;
    int weight_shape[]   = {normalized_shape};
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
