#include "nn/layers/batchnorm1d.h"
#include "nn/layers/batchnorm2d.h"
#include "nn/layers/batchnorm3d.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <string.h>
#include "alloc/cml_allocator.h"

/* Deepest input rank the normalisation layers accept (BatchNorm3d, [N,C,D,H,W]). */
#define NORM_MAX_NDIM 5

/* Fill `out` with `shape` collapsed to 1 everywhere but `channel_dim`, the view
 * a 1-D per-channel statistic must take before it can broadcast over `shape`. */
static void channel_broadcast_shape(int* out, const int* shape, int ndim, int channel_dim) {
    for (int i = 0; i < ndim; i++)
        out[i] = (i == channel_dim) ? shape[channel_dim] : 1;
}

Tensor* nn_norm_affine(Tensor* x, const Parameter* weight, const Parameter* bias, const int* shape,
                       int ndim, int channel_dim) {
    if (!x || !weight || !bias || ndim > NORM_MAX_NDIM)
        return x;

    int stat_shape[NORM_MAX_NDIM];
    channel_broadcast_shape(stat_shape, shape, ndim, channel_dim);

    Tensor* gamma = uop_expand_to(uop_reshape_to(weight->tensor, stat_shape, ndim), shape, ndim);
    Tensor* beta  = uop_expand_to(uop_reshape_to(bias->tensor, stat_shape, ndim), shape, ndim);
    return uop_add(uop_mul(gamma, x), beta);
}

Tensor* nn_norm_rowwise(Tensor* x, int rows, int cols, float eps) {
    if (!x)
        return NULL;

    int flat[] = {rows, cols};
    int stat[] = {rows, 1};

    Tensor* x2   = uop_reshape_to(x, flat, 2);
    Tensor* mean = uop_reshape_to(uop_mean_dim(x2, 1, false), stat, 2);
    Tensor* diff = uop_sub(x2, uop_expand_to(mean, flat, 2));
    Tensor* var  = uop_reshape_to(uop_mean_dim(uop_mul(diff, diff), 1, false), stat, 2);

    TensorConfig config = {
        .dtype = x->dtype, .device = x->device, .has_dtype = true, .has_device = true};
    Tensor* eps_tensor = tensor_full(stat, 2, &config, eps);
    if (!eps_tensor)
        return NULL;

    Tensor* std = uop_sqrt(uop_add(var, eps_tensor));
    tensor_free(eps_tensor);

    return uop_div(diff, uop_expand_to(std, flat, 2));
}

/* Recompute current_mean/current_var from this batch and fold them into the
 * running statistics. `input` is viewed as [batch, channels, spatial]; the
 * per-channel mean is taken over the spatial axis and then over the batch,
 * because uop_mean reduces a single axis at a time. */
static int batchnorm_update_stats(BatchNormState* bn, Tensor* input, int batch, int channels,
                                  int spatial) {
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    if (!bn->current_mean) {
        int stat_shape[] = {channels};
        bn->current_mean = tensor_zeros(stat_shape, 1, &config);
        bn->current_var  = tensor_zeros(stat_shape, 1, &config);
        if (!bn->current_mean || !bn->current_var)
            return -1;
    }

    int grouped[]  = {batch, channels, spatial};
    int per_chan[] = {1, channels, 1};
    Tensor* x3     = uop_reshape_to(input, grouped, 3);
    Tensor* mean   = uop_mean_dim(uop_mean_dim(x3, 2, false), 0, false);

    float* mean_data         = (float*)tensor_data_ptr(mean);
    float* current_mean_data = (float*)tensor_data_ptr(bn->current_mean);
    if (!mean_data || !current_mean_data)
        return -1;
    memcpy(current_mean_data, mean_data, (size_t)channels * sizeof(float));

    /* Centre against the materialised copy, not `mean`, so the statistics stay
     * out of the gradient graph (as PyTorch's running stats do). */
    Tensor* mean_broadcast =
        uop_expand_to(uop_reshape_to(bn->current_mean, per_chan, 3), grouped, 3);
    Tensor* diff = uop_sub(x3, mean_broadcast);
    Tensor* var  = uop_mean_dim(uop_mean_dim(uop_mul(diff, diff), 2, false), 0, false);

    float* var_data         = (float*)tensor_data_ptr(var);
    float* current_var_data = (float*)tensor_data_ptr(bn->current_var);
    if (!var_data || !current_var_data)
        return -1;
    memcpy(current_var_data, var_data, (size_t)channels * sizeof(float));

    if (bn->track_running_stats && bn->running_mean && bn->running_var) {
        float* running_mean = (float*)tensor_data_ptr(bn->running_mean);
        float* running_var  = (float*)tensor_data_ptr(bn->running_var);
        if (running_mean && running_var) {
            for (int c = 0; c < channels; c++) {
                running_mean[c] =
                    bn->momentum * running_mean[c] + (1.0f - bn->momentum) * current_mean_data[c];
                running_var[c] =
                    bn->momentum * running_var[c] + (1.0f - bn->momentum) * current_var_data[c];
            }
        }
    }

    return 0;
}

static Tensor* batchnorm_forward(Module* module, Tensor* input) {
    BatchNormState* bn = (BatchNormState*)module;
    if (!bn || !input)
        return NULL;

    if (input->ndim != bn->input_ndim) {
        LOG_ERROR("%s expects %dD input [N, C, ...], got %dD", module->name, bn->input_ndim,
                  input->ndim);
        return NULL;
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    if (channels != bn->num_features) {
        LOG_ERROR("%s: input channels (%d) doesn't match num_features (%d)", module->name, channels,
                  bn->num_features);
        return NULL;
    }

    int spatial = 1;
    for (int i = 2; i < input->ndim; i++)
        spatial *= input->shape[i];

    bool training = module_is_training(module);

    int stat_shape[NORM_MAX_NDIM];
    channel_broadcast_shape(stat_shape, input->shape, input->ndim, 1);
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};

    Tensor* centered      = NULL;
    Tensor* std_broadcast = NULL;

    if (training) {
        /* Update running stats as a detached side effect (for eval), but compute
         * the normalization's mean/var as differentiable uops of the input so the
         * backward flows through the batch statistics -- treating them as
         * constants gave a wrong (mean/var-invariance-violating) gradient.
         * Reduce over [N, spatial] per channel via the proven single-dim keepdim
         * reduces on a [N,C,S] view (multi-dim reduce keepdim mis-shapes here). */
        if (batchnorm_update_stats(bn, input, batch, channels, spatial) != 0)
            return NULL;

        int grouped[]     = {batch, channels, spatial};
        Tensor* x3        = uop_reshape_to(input, grouped, 3);
        Tensor* mean      = uop_mean_dim(uop_mean_dim(x3, 2, true), 0, true); /* [1,C,1] */
        Tensor* centered3 = uop_sub(x3, mean);
        Tensor* var = uop_mean_dim(uop_mean_dim(uop_mul(centered3, centered3), 2, true), 0, true);

        int one_shape[]    = {1};
        Tensor* eps_tensor = tensor_full(one_shape, 1, &config, bn->eps);
        if (!eps_tensor)
            return NULL;
        Tensor* std3 = uop_sqrt(uop_add(var, eps_tensor)); /* [1,C,1] */
        tensor_free(eps_tensor);

        Tensor* normalized3 = uop_div(centered3, std3);
        Tensor* normalized  = uop_reshape_to(normalized3, input->shape, input->ndim);
        if (!normalized)
            return NULL;
        return nn_norm_affine(normalized, bn->weight, bn->bias, input->shape, input->ndim, 1);
    } else {
        Tensor* mean_tensor = bn->running_mean;
        Tensor* var_tensor  = bn->running_var;
        if (!mean_tensor || !var_tensor) {
            LOG_ERROR("%s: missing mean or variance tensor", module->name);
            return NULL;
        }
        Tensor* mean_broadcast = uop_expand_to(uop_reshape_to(mean_tensor, stat_shape, input->ndim),
                                               input->shape, input->ndim);
        centered               = uop_sub(input, mean_broadcast);

        int channel_shape[] = {channels};
        Tensor* eps_tensor  = tensor_full(channel_shape, 1, &config, bn->eps);
        if (!eps_tensor)
            return NULL;
        Tensor* std = uop_sqrt(uop_add(var_tensor, eps_tensor));
        tensor_free(eps_tensor);
        std_broadcast =
            uop_expand_to(uop_reshape_to(std, stat_shape, input->ndim), input->shape, input->ndim);
    }

    Tensor* normalized = uop_div(centered, std_broadcast);
    if (!normalized)
        return NULL;

    return nn_norm_affine(normalized, bn->weight, bn->bias, input->shape, input->ndim, 1);
}

static void batchnorm_free(Module* module) {
    BatchNormState* bn = (BatchNormState*)module;
    if (!bn)
        return;
    if (bn->running_mean)
        tensor_free(bn->running_mean);
    if (bn->running_var)
        tensor_free(bn->running_var);
    if (bn->current_mean)
        tensor_free(bn->current_mean);
    if (bn->current_var)
        tensor_free(bn->current_var);
    cml_free(bn);
}

BatchNormState* nn_batchnorm_new(const char* name, int input_ndim, int num_features, float eps,
                                 float momentum, bool affine, bool track_running_stats, DType dtype,
                                 DeviceType device) {
    BatchNormState* bn = cml_malloc(sizeof(BatchNormState));
    if (!bn)
        return NULL;

    if (module_init((Module*)bn, name, batchnorm_forward, batchnorm_free) != 0) {
        cml_free(bn);
        return NULL;
    }

    bn->num_features        = num_features;
    bn->input_ndim          = input_ndim;
    bn->eps                 = eps;
    bn->momentum            = momentum;
    bn->affine              = affine;
    bn->track_running_stats = track_running_stats;
    bn->current_mean        = NULL;
    bn->current_var         = NULL;

    if (affine) {
        if (nn_add_affine_params((Module*)bn, num_features, dtype, device, &bn->weight,
                                 &bn->bias) != 0)
            return NULL;
    } else {
        bn->weight = NULL;
        bn->bias   = NULL;
    }

    if (track_running_stats) {
        if (nn_add_running_stats((Module*)bn, num_features, dtype, device, &bn->running_mean,
                                 &bn->running_var) != 0)
            return NULL;
        /* Register in the module's buffer registry so DDP can broadcast the
         * running stats from rank 0 at forward time (broadcast_buffers). */
        module_add_buffer((Module*)bn, bn->running_mean, "running_mean");
        module_add_buffer((Module*)bn, bn->running_var, "running_var");
    } else {
        bn->running_mean = NULL;
        bn->running_var  = NULL;
    }

    return bn;
}

BatchNorm1d* nn_batchnorm1d(int num_features, float eps, float momentum, bool affine,
                            bool track_running_stats, DType dtype, DeviceType device) {
    return nn_batchnorm_new("BatchNorm1d", 2, num_features, eps, momentum, affine,
                            track_running_stats, dtype, device);
}

BatchNorm2d* nn_batchnorm2d(int num_features, float eps, float momentum, bool affine,
                            bool track_running_stats, DType dtype, DeviceType device) {
    return nn_batchnorm_new("BatchNorm2d", 4, num_features, eps, momentum, affine,
                            track_running_stats, dtype, device);
}

BatchNorm3d* nn_batchnorm3d(int num_features, float eps, float momentum, bool affine,
                            bool track_running_stats, DType dtype, DeviceType device) {
    return nn_batchnorm_new("BatchNorm3d", 5, num_features, eps, momentum, affine,
                            track_running_stats, dtype, device);
}
