#include "nn/layers/prelu.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include <stdlib.h>
#include "alloc/cml_allocator.h"

/** torch.nn.PReLU forward: max(0,x) + alpha * min(0,x), with the learnable alpha broadcast
 *  over the input shape. Returns NULL on NULL args or a missing alpha parameter. */
static Tensor* prelu_forward(Module* module, Tensor* input) {
    PReLU* prelu = (PReLU*)module;
    if (!prelu || !input || !prelu->alpha || !prelu->alpha->tensor)
        return NULL;

    Tensor* pos = uop_relu(input);
    if (!pos)
        return NULL;

    Tensor* neg_part = uop_relu(uop_neg(input));
    if (!neg_part)
        return NULL;

    ExpandParams expand_params = {.new_shape = input->shape, .new_ndim = input->ndim};
    Tensor* alpha_broadcast    = uop_expand(prelu->alpha->tensor, &expand_params);
    if (!alpha_broadcast)
        return NULL;

    Tensor* scaled_neg = uop_mul(alpha_broadcast, neg_part);
    if (!scaled_neg)
        return NULL;

    return uop_sub(pos, scaled_neg);
}

/** Free the PReLU module; the alpha parameter is released by module_free. */
static void prelu_free(Module* module) { cml_free(module); }

/** Construct a PReLU layer with num_parameters alpha slopes (clamped to >=1) initialized to
 *  init (defaults to 0.25 when 0). Returns NULL on failure. */
PReLU* nn_prelu(int num_parameters, float init, DType dtype, DeviceType device) {
    if (num_parameters <= 0)
        num_parameters = 1;
    if (init == 0.0f)
        init = 0.25f;

    PReLU* prelu = cml_malloc(sizeof(PReLU));
    if (!prelu)
        return NULL;

    if (module_init((Module*)prelu, "PReLU", prelu_forward, prelu_free) != 0) {
        cml_free(prelu);
        return NULL;
    }

    prelu->num_parameters_ = num_parameters;

    int param_shape[] = {num_parameters};
    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* alpha = tensor_full(param_shape, 1, &config, init);
    if (!alpha) {
        module_free((Module*)prelu);
        return NULL;
    }

    if (module_add_parameter((Module*)prelu, alpha, "alpha", true) != 0) {
        tensor_free(alpha);
        module_free((Module*)prelu);
        return NULL;
    }
    prelu->alpha = module_get_parameter((Module*)prelu, "alpha");
    return prelu;
}
