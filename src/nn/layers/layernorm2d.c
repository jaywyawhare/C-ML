#include "nn/layers/layernorm2d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"

static Tensor* layernorm2d_forward(Module* module, Tensor* input) {
    LayerNorm2d* ln = (LayerNorm2d*)module;

    if (!ln || !input)
        return NULL;
    if (input->ndim != 4) {
        LOG_ERROR("LayerNorm2d expects 4D input [N, C, H, W], got %dD", input->ndim);
        return NULL;
    }

    int channels = input->shape[1];
    if (channels != ln->num_channels) {
        LOG_ERROR("LayerNorm2d: input channels (%d) doesn't match num_channels (%d)", channels,
                  ln->num_channels);
        return NULL;
    }

    /* One statistic per sample: rows are N, columns the whole C*H*W volume. */
    Tensor* normalized = nn_norm_rowwise(input, input->shape[0],
                                         channels * input->shape[2] * input->shape[3], ln->eps);
    Tensor* output     = uop_reshape_to(normalized, input->shape, 4);
    if (!output)
        return NULL;

    return nn_norm_affine(output, ln->weight, ln->bias, input->shape, 4, 1);
}

static void layernorm2d_free(Module* module) {
    if (module)
        cml_free(module);
}

LayerNorm2d* nn_layernorm2d(int num_channels, float eps, bool affine, DType dtype,
                            DeviceType device) {
    LayerNorm2d* ln = cml_malloc(sizeof(LayerNorm2d));
    if (!ln)
        return NULL;

    if (module_init((Module*)ln, "LayerNorm2d", layernorm2d_forward, layernorm2d_free) != 0) {
        cml_free(ln);
        return NULL;
    }

    ln->num_channels = num_channels;
    ln->eps          = eps > 0.0f ? eps : 1e-5f;
    ln->affine       = affine;

    if (affine) {
        if (nn_add_affine_params((Module*)ln, num_channels, dtype, device,
                                 &ln->weight, &ln->bias) != 0)
            return NULL;
    } else {
        ln->weight = NULL;
        ln->bias   = NULL;
    }

    return ln;
}
