#include "nn/layers/layernorm2d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>

static Tensor* layernorm2d_forward(Module* module, Tensor* input) {
    LayerNorm2d* ln = (LayerNorm2d*)module;

    if (!ln || !input)
        return NULL;

    if (input->ndim != 4) {
        LOG_ERROR("LayerNorm2d expects 4D input [N, C, H, W], got %dD", input->ndim);
        return NULL;
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    if (channels != ln->num_channels) {
        LOG_ERROR("LayerNorm2d: input channels (%d) doesn't match num_channels (%d)", channels,
                  ln->num_channels);
        return NULL;
    }

    int last = channels * height * width;
    ReshapeParams rp;
    int rs[]     = {batch, last};
    rp.new_shape = rs;
    rp.new_ndim  = 2;

    Tensor* x2 = uop_reshape(input, &rp);
    if (!x2)
        return NULL;

    ReduceParams mp;
    int md[]    = {1};
    mp.dims     = md;
    mp.num_dims = 1;
    mp.keepdim  = false;

    Tensor* mean_1d = uop_mean(x2, &mp);
    if (!mean_1d)
        return NULL;

    int mean2[] = {batch, 1};
    ReshapeParams rmean = {.new_shape = mean2, .new_ndim = 2};
    Tensor* mean_r      = uop_reshape(mean_1d, &rmean);
    if (!mean_r)
        return NULL;

    ExpandParams ep;
    ep.new_shape = rs;
    ep.new_ndim  = 2;

    Tensor* mean_b = uop_expand(mean_r, &ep);
    if (!mean_b)
        return NULL;

    Tensor* diff = uop_sub(x2, mean_b);
    if (!diff)
        return NULL;

    Tensor* dsq = uop_mul(diff, diff);
    if (!dsq)
        return NULL;

    Tensor* var_1d = uop_mean(dsq, &mp);
    if (!var_1d)
        return NULL;

    Tensor* var_r = uop_reshape(var_1d, &rmean);
    if (!var_r)
        return NULL;

    TensorConfig cfg = {.dtype      = input->dtype,
                        .device     = input->device,
                        .has_dtype  = true,
                        .has_device = true};
    int eps_sh[]  = {batch, 1};
    Tensor* eps_t = tensor_full(eps_sh, 2, &cfg, ln->eps);
    if (!eps_t)
        return NULL;

    Tensor* ve = uop_add(var_r, eps_t);
    if (!ve)
        return NULL;

    Tensor* std = uop_sqrt(ve);
    if (!std)
        return NULL;

    Tensor* std_b = uop_expand(std, &ep);
    if (!std_b)
        return NULL;

    Tensor* norm2 = uop_div(diff, std_b);
    if (!norm2)
        return NULL;

    int out4[] = {batch, channels, height, width};
    ReshapeParams rout = {.new_shape = out4, .new_ndim = 4};
    Tensor* output     = uop_reshape(norm2, &rout);
    if (!output)
        return NULL;

    if (ln->affine && ln->weight && ln->bias) {
        ReshapeParams rstat;
        int st[]        = {1, channels, 1, 1};
        rstat.new_shape = st;
        rstat.new_ndim  = 4;

        Tensor* w4 = uop_reshape(ln->weight->tensor, &rstat);
        Tensor* b4 = uop_reshape(ln->bias->tensor, &rstat);
        if (!w4 || !b4)
            return NULL;

        ExpandParams ex;
        ex.new_ndim  = 4;
        ex.new_shape = out4;
        Tensor* wb   = uop_expand(w4, &ex);
        Tensor* bb   = uop_expand(b4, &ex);
        if (!wb || !bb)
            return NULL;

        Tensor* sc = uop_mul(wb, output);
        if (!sc)
            return NULL;
        output = uop_add(sc, bb);
        if (!output)
            return NULL;
    }

    return output;
}

static void layernorm2d_free(Module* module) {
    LayerNorm2d* ln = (LayerNorm2d*)module;
    if (!ln)
        return;
    free(ln);
}

LayerNorm2d* nn_layernorm2d(int num_channels, float eps, bool affine, DType dtype,
                            DeviceType device) {
    LayerNorm2d* ln = malloc(sizeof(LayerNorm2d));
    if (!ln)
        return NULL;

    if (module_init((Module*)ln, "LayerNorm2d", layernorm2d_forward, layernorm2d_free) != 0) {
        free(ln);
        return NULL;
    }

    ln->num_channels = num_channels;
    ln->eps          = eps > 0.0f ? eps : 1e-5f;
    ln->affine       = affine;

    TensorConfig config =
        (TensorConfig){.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

    if (affine) {
        int param_shape[] = {num_channels};
        Tensor* weight    = tensor_ones(param_shape, 1, &config);
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

        Tensor* bias = tensor_zeros(param_shape, 1, &config);
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
