#include "nn/layers/instancenorm.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"

static Tensor* instancenorm2d_forward(Module* module, Tensor* input) {
    InstanceNorm2d* in = (InstanceNorm2d*)module;

    if (!in || !input)
        return NULL;
    if (input->ndim != 4) {
        LOG_ERROR("InstanceNorm2d expects 4D input [N, C, H, W], got %dD", input->ndim);
        return NULL;
    }

    int channels = input->shape[1];
    if (channels != in->num_features) {
        LOG_ERROR("InstanceNorm2d: input channels (%d) doesn't match num_features (%d)", channels,
                  in->num_features);
        return NULL;
    }

    /* One statistic per (sample, channel): rows are N*C, columns the H*W plane. */
    Tensor* normalized = nn_norm_rowwise(input, input->shape[0] * channels,
                                         input->shape[2] * input->shape[3], in->eps);
    Tensor* output     = uop_reshape_to(normalized, input->shape, 4);
    if (!output)
        return NULL;

    return nn_norm_affine(output, in->weight, in->bias, input->shape, 4, 1);
}

static void instancenorm2d_free(Module* module) {
    if (module)
        cml_free(module);
}

InstanceNorm2d* nn_instancenorm2d(int num_features, float eps, bool affine, DType dtype,
                                  DeviceType device) {
    InstanceNorm2d* in = cml_malloc(sizeof(InstanceNorm2d));
    if (!in)
        return NULL;

    if (module_init((Module*)in, "InstanceNorm2d", instancenorm2d_forward, instancenorm2d_free) !=
        0) {
        cml_free(in);
        return NULL;
    }

    in->num_features = num_features;
    in->eps          = eps > 0.0f ? eps : 1e-5f;
    in->affine       = affine;

    if (affine) {
        if (nn_add_affine_params((Module*)in, num_features, dtype, device,
                                 &in->weight, &in->bias) != 0)
            return NULL;
    } else {
        in->weight = NULL;
        in->bias   = NULL;
    }

    return in;
}
