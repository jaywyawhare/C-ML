#include "autograd/amp.h"
#include "autograd/autograd.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

static AutocastContext g_autocast_ctx = {
    .enabled      = false,
    .target_dtype = DTYPE_FLOAT16
};

void autocast_enter(DType target_dtype) {
    g_autocast_ctx.enabled      = true;
    g_autocast_ctx.target_dtype = target_dtype;
}

void autocast_exit(void) {
    g_autocast_ctx.enabled = false;
}

bool autocast_is_enabled(void) {
    return g_autocast_ctx.enabled;
}

AutocastContext* autocast_get_context(void) {
    return &g_autocast_ctx;
}

bool autocast_should_keep_float32(OpType op) {
    switch (op) {
        /* Numerically sensitive operations that need full precision */
        case OP_SOFTMAX:
        case OP_LOG_SOFTMAX:
        case OP_MSE_LOSS:
        case OP_MAE_LOSS:
        case OP_BCE_LOSS:
        case OP_CROSS_ENTROPY_LOSS:
        case OP_HUBER_LOSS:
        case OP_KL_DIV_LOSS:
        case OP_LOG:
        case OP_EXP:
        case OP_POW:
            return true;
        default:
            return false;
    }
}

GradScaler* grad_scaler_create(float init_scale, float growth_factor,
                                float backoff_factor, int growth_interval) {
    GradScaler* scaler = calloc(1, sizeof(GradScaler));
    if (!scaler) {
        LOG_ERROR("GradScaler: failed to allocate memory");
        return NULL;
    }

    scaler->scale_factor    = (init_scale > 0.0f) ? init_scale : 65536.0f;
    scaler->growth_factor   = (growth_factor > 0.0f) ? growth_factor : 2.0f;
    scaler->backoff_factor  = (backoff_factor > 0.0f) ? backoff_factor : 0.5f;
    scaler->growth_interval = (growth_interval > 0) ? growth_interval : 2000;
    scaler->growth_step     = 0;
    scaler->found_inf       = false;

    return scaler;
}

void grad_scaler_free(GradScaler* scaler) {
    if (!scaler)
        return;
    free(scaler);
}

Tensor* grad_scaler_scale(GradScaler* scaler, Tensor* loss) {
    if (!scaler || !loss) {
        LOG_ERROR("grad_scaler_scale: NULL argument");
        return NULL;
    }

    tensor_ensure_executed(loss);

    int scalar_shape[] = {1};
    TensorConfig config = (TensorConfig){
        .dtype = loss->dtype, .device = loss->device, .has_dtype = true, .has_device = true};
    Tensor* scale_tensor = tensor_full(scalar_shape, 1, &config, scaler->scale_factor);
    if (!scale_tensor) {
        LOG_ERROR("grad_scaler_scale: failed to create scale tensor");
        return NULL;
    }

    tensor_ensure_executed(scale_tensor);

    Tensor* scaled = tensor_empty(loss->shape, loss->ndim, &config);
    if (!scaled) {
        tensor_free(scale_tensor);
        return NULL;
    }

    float* loss_data   = (float*)loss->data;
    float* scaled_data = (float*)scaled->data;
    float sf           = scaler->scale_factor;

    if (!loss_data || !scaled_data) {
        tensor_free(scaled);
        tensor_free(scale_tensor);
        return NULL;
    }

    for (size_t i = 0; i < loss->numel; i++) {
        scaled_data[i] = loss_data[i] * sf;
    }

    tensor_free(scale_tensor);

    return scaled;
}

void grad_scaler_unscale(GradScaler* scaler, Parameter** params, int num_params) {
    if (!scaler || !params) {
        LOG_ERROR("grad_scaler_unscale: NULL argument");
        return;
    }

    scaler->found_inf = false;
    float inv_scale = 1.0f / scaler->scale_factor;

    for (int p = 0; p < num_params; p++) {
        if (!params[p] || !params[p]->tensor)
            continue;

        Tensor* grad = params[p]->tensor->grad;
        if (!grad)
            continue;

        tensor_ensure_executed(grad);
        float* grad_data = (float*)grad->data;
        if (!grad_data)
            continue;

        for (size_t i = 0; i < grad->numel; i++) {
            grad_data[i] *= inv_scale;

            if (isinf(grad_data[i]) || isnan(grad_data[i])) {
                scaler->found_inf = true;
            }
        }
    }

    if (scaler->found_inf) {
        LOG_WARNING("GradScaler: inf/nan detected in gradients, will skip optimizer step");
    }
}

void grad_scaler_step(GradScaler* scaler, void (*step_fn)(void*), void* optimizer) {
    if (!scaler || !step_fn) {
        LOG_ERROR("grad_scaler_step: NULL argument");
        return;
    }

    if (scaler->found_inf) {
        LOG_INFO("GradScaler: skipping optimizer step due to inf/nan gradients");
        return;
    }

    step_fn(optimizer);
}

void grad_scaler_update(GradScaler* scaler) {
    if (!scaler)
        return;

    if (scaler->found_inf) {
        scaler->scale_factor *= scaler->backoff_factor;
        scaler->growth_step = 0;

        if (scaler->scale_factor < 1.0f) {
            scaler->scale_factor = 1.0f;
        }

        LOG_INFO("GradScaler: reduced scale to %.1f after inf/nan", scaler->scale_factor);
    } else {
        scaler->growth_step++;

        if (scaler->growth_step >= scaler->growth_interval) {
            scaler->scale_factor *= scaler->growth_factor;
            scaler->growth_step = 0;

            if (scaler->scale_factor > 65536.0f * 65536.0f) {
                scaler->scale_factor = 65536.0f * 65536.0f;
            }

        }
    }

    scaler->found_inf = false;
}
