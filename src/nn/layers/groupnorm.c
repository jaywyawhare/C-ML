#include "nn/layers/groupnorm.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <math.h>

static Tensor* groupnorm_forward(Module* module, Tensor* input) {
    GroupNorm* gn = (GroupNorm*)module;

    if (!gn || !input)
        return NULL;

    if (input->ndim < 2) {
        LOG_ERROR("GroupNorm expects at least 2D input, got %dD", input->ndim);
        return NULL;
    }

    int C = input->shape[1];
    if (C != gn->num_channels) {
        LOG_ERROR("GroupNorm: input channels (%d) doesn't match num_channels (%d)", C,
                  gn->num_channels);
        return NULL;
    }

    int N = input->shape[0];
    int G = gn->num_groups;
    int channels_per_group = C / G;

    int spatial = 1;
    for (int i = 2; i < input->ndim; i++)
        spatial *= input->shape[i];

    int group_size = channels_per_group * spatial;

    ReshapeParams rp;
    int rs[]         = {N * G, group_size};
    rp.new_shape     = rs;
    rp.new_ndim      = 2;
    Tensor* x2       = uop_reshape(input, &rp);
    if (!x2)
        return NULL;

    ReduceParams mean_params;
    int mean_dims[]      = {1};
    mean_params.dims     = mean_dims;
    mean_params.num_dims = 1;
    mean_params.keepdim  = false;

    Tensor* mean_1d = uop_mean(x2, &mean_params);
    if (!mean_1d)
        return NULL;

    int mean2_shape[] = {N * G, 1};
    ReshapeParams rmean = {.new_shape = mean2_shape, .new_ndim = 2};
    Tensor* mean_reduced = uop_reshape(mean_1d, &rmean);
    if (!mean_reduced)
        return NULL;

    ExpandParams expand_mv;
    expand_mv.new_shape = rs;
    expand_mv.new_ndim  = 2;

    Tensor* mean_broadcast = uop_expand(mean_reduced, &expand_mv);
    if (!mean_broadcast)
        return NULL;

    Tensor* diff = uop_sub(x2, mean_broadcast);
    if (!diff)
        return NULL;

    Tensor* diff_sq = uop_mul(diff, diff);
    if (!diff_sq)
        return NULL;

    Tensor* var_1d = uop_mean(diff_sq, &mean_params);
    if (!var_1d)
        return NULL;

    Tensor* var_reduced = uop_reshape(var_1d, &rmean);
    if (!var_reduced)
        return NULL;

    TensorConfig cfg = {.dtype      = input->dtype,
                        .device     = input->device,
                        .has_dtype  = true,
                        .has_device = true};
    int eps_sh[]  = {N * G, 1};
    Tensor* eps_t = tensor_full(eps_sh, 2, &cfg, gn->eps);
    if (!eps_t)
        return NULL;

    Tensor* var_eps = uop_add(var_reduced, eps_t);
    if (!var_eps)
        return NULL;

    Tensor* std_tensor = uop_sqrt(var_eps);
    if (!std_tensor)
        return NULL;

    Tensor* std_broadcast = uop_expand(std_tensor, &expand_mv);
    if (!std_broadcast)
        return NULL;

    Tensor* normalized2 = uop_div(diff, std_broadcast);
    if (!normalized2)
        return NULL;

    int* back_shape = tensor_shape_copy(input->shape, input->ndim);
    if (!back_shape)
        return NULL;
    ReshapeParams rback = {.new_shape = back_shape, .new_ndim = input->ndim};
    Tensor* output      = uop_reshape(normalized2, &rback);
    free(back_shape);
    if (!output)
        return NULL;

    if (gn->affine && gn->weight && gn->bias) {
        int nd = input->ndim;
        int* stat_shape = malloc((size_t)nd * sizeof(int));
        if (!stat_shape)
            return NULL;
        stat_shape[0] = 1;
        stat_shape[1] = C;
        for (int i = 2; i < nd; i++)
            stat_shape[i] = 1;

        ReshapeParams rw = {.new_shape = stat_shape, .new_ndim = nd};
        Tensor* w_r      = uop_reshape(gn->weight->tensor, &rw);
        Tensor* b_r      = uop_reshape(gn->bias->tensor, &rw);
        free(stat_shape);
        if (!w_r || !b_r)
            return NULL;

        ExpandParams ex;
        ex.new_ndim  = nd;
        ex.new_shape = tensor_shape_copy(input->shape, nd);
        if (!ex.new_shape)
            return NULL;

        Tensor* wb = uop_expand(w_r, &ex);
        Tensor* bb = uop_expand(b_r, &ex);
        free(ex.new_shape);
        if (!wb || !bb)
            return NULL;

        Tensor* scaled = uop_mul(wb, output);
        if (!scaled)
            return NULL;
        output = uop_add(scaled, bb);
        if (!output)
            return NULL;
    }

    return output;
}

static void groupnorm_free(Module* module) { free(module); }

GroupNorm* nn_groupnorm(int num_groups, int num_channels, float eps, bool affine,
                        DType dtype, DeviceType device) {
    if (num_channels % num_groups != 0) {
        LOG_ERROR("num_channels (%d) must be divisible by num_groups (%d)", num_channels,
                  num_groups);
        return NULL;
    }

    GroupNorm* gn = malloc(sizeof(GroupNorm));
    if (!gn)
        return NULL;

    if (module_init((Module*)gn, "GroupNorm", groupnorm_forward, groupnorm_free) != 0) {
        free(gn);
        return NULL;
    }

    gn->num_groups   = num_groups;
    gn->num_channels = num_channels;
    gn->eps          = eps > 0 ? eps : 1e-5f;
    gn->affine       = affine;

    if (affine) {
        int param_shape[] = {num_channels};
        TensorConfig config = (TensorConfig){
            .dtype = dtype, .device = device, .has_dtype = true, .has_device = true};

        Tensor* weight = tensor_ones(param_shape, 1, &config);
        if (!weight) {
            module_free((Module*)gn);
            return NULL;
        }

        if (module_add_parameter((Module*)gn, weight, "weight", true) != 0) {
            tensor_free(weight);
            module_free((Module*)gn);
            return NULL;
        }

        gn->weight = module_get_parameter((Module*)gn, "weight");

        Tensor* bias = tensor_zeros(param_shape, 1, &config);
        if (!bias) {
            module_free((Module*)gn);
            return NULL;
        }

        if (module_add_parameter((Module*)gn, bias, "bias", true) != 0) {
            tensor_free(bias);
            module_free((Module*)gn);
            return NULL;
        }

        gn->bias = module_get_parameter((Module*)gn, "bias");
    } else {
        gn->weight = NULL;
        gn->bias   = NULL;
    }

    return gn;
}
