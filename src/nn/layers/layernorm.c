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
#include "alloc/cml_allocator.h"

/** torch.nn.LayerNorm forward: normalize over the last dim to zero mean/unit variance (with
 *  eps), then apply the optional affine weight/bias. NULL on rank or last-dim mismatch. */
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
    /* One fused kernel computes mean/var/normalize/affine over the last dim in a
     * single pass (and one fused backward), instead of a ~12-op primitive chain. */
    Tensor* gamma = (ln->affine && ln->weight) ? ln->weight->tensor : NULL;
    Tensor* beta  = (ln->affine && ln->bias) ? ln->bias->tensor : NULL;

    Tensor* output = uop_layernorm(input, gamma, beta, ln->eps);
    if (!output)
        return NULL;

    if (autograd_is_grad_enabled() && input->requires_grad) {
        output->requires_grad = true;
    }

    return output;
}

/** Free the LayerNorm module; affine parameters are released by module_free. */
static void layernorm_free(Module* module) {
    LayerNorm* ln = (LayerNorm*)module;
    if (!ln)
        return;

    module_free(module);
}

/** Construct a LayerNorm over a last dim of size normalized_shape; allocates weight/bias when
 *  affine. eps falls back to 1e-5 if non-positive. NULL on bad shape or failure. */
LayerNorm* nn_layernorm(int normalized_shape, float eps, bool affine, DType dtype,
                        DeviceType device) {
    if (normalized_shape <= 0) {
        LOG_ERROR("LayerNorm: normalized_shape must be positive, got %d", normalized_shape);
        return NULL;
    }

    LayerNorm* ln = cml_malloc(sizeof(LayerNorm));
    if (!ln)
        return NULL;
    if (module_init((Module*)ln, "LayerNorm", layernorm_forward, layernorm_free) != 0) {
        cml_free(ln);
        return NULL;
    }
    ln->normalized_shape = normalized_shape;
    ln->eps              = eps > 0.0f ? eps : 1e-5f;
    ln->affine           = affine;
    ln->weight           = NULL;
    ln->bias             = NULL;
    if (affine) {
        if (nn_add_affine_params((Module*)ln, normalized_shape, dtype, device, &ln->weight,
                                 &ln->bias) != 0)
            return NULL;
    } else {
        ln->weight = NULL;
        ln->bias   = NULL;
    }

    return ln;
}
