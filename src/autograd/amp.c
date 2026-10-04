#include "autograd/amp.h"
#include "autograd/autograd.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "alloc/cml_allocator.h"

static AutocastContext g_autocast_ctx = {.enabled = false, .target_dtype = DTYPE_FLOAT16};

/* Target-dtype selector: read once from CML_AMP_DTYPE ("bf16"/"bfloat16"
 * selects BFLOAT16), cached like GRAD_MODE in autodiff.c. Default stays
 * FLOAT16 for backward compatibility. */
static int autocast_env_dtype(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* m = getenv("CML_AMP_DTYPE");
        cached = (m && (strcmp(m, "bf16") == 0 || strcmp(m, "bfloat16") == 0)) ? DTYPE_BFLOAT16
                                                                               : DTYPE_FLOAT16;
    }
    return cached;
}

static bool g_dtype_resolved = false;

/** Latch the env-selected target dtype into the global context once, on first use. */
static void autocast_resolve_dtype(void) {
    if (!g_dtype_resolved) {
        g_autocast_ctx.target_dtype = (DType)autocast_env_dtype();
        g_dtype_resolved            = true;
    }
}

/** Enable autocast and pin the target low-precision dtype for the enclosing region. */
void autocast_enter(DType target_dtype) {
    autocast_resolve_dtype();
    g_autocast_ctx.enabled      = true;
    g_autocast_ctx.target_dtype = target_dtype;
    g_dtype_resolved            = true;
}

/** Leave the autocast region; the pinned target dtype is left untouched. */
void autocast_exit(void) { g_autocast_ctx.enabled = false; }

/** Whether ops should currently be cast to the autocast target dtype. */
bool autocast_is_enabled(void) { return g_autocast_ctx.enabled; }

/** Return the shared autocast context, resolving the env dtype first. */
AutocastContext* autocast_get_context(void) {
    autocast_resolve_dtype();
    return &g_autocast_ctx;
}

/** Target dtype selected by CML_AMP_DTYPE, independent of whether autocast is on. */
DType autocast_default_dtype(void) { return (DType)autocast_env_dtype(); }

/** Override the autocast target dtype, marking it resolved so env lookup is skipped. */
void autocast_set_dtype(DType dtype) {
    g_autocast_ctx.target_dtype = dtype;
    g_dtype_resolved            = true;
}

/** Current autocast target dtype, resolving the env default on first access. */
DType autocast_get_dtype(void) {
    autocast_resolve_dtype();
    return g_autocast_ctx.target_dtype;
}

/** True for numerically sensitive ops (softmax, losses, log/exp/pow) that must stay fp32
 * even under autocast to avoid overflow or loss of precision. */
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

/** Allocate a loss-scale controller; non-positive arguments fall back to defaults
 * (scale 65536, growth 2x, backoff 0.5x, interval 2000). Returns NULL on OOM. */
GradScaler* grad_scaler_create(float init_scale, float growth_factor, float backoff_factor,
                               int growth_interval) {
    GradScaler* scaler = cml_calloc(1, sizeof(GradScaler));
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

/** Free a GradScaler; NULL-safe. */
void grad_scaler_free(GradScaler* scaler) {
    if (!scaler)
        return;
    cml_free(scaler);
}

/** Multiply the loss by the current scale so fp16 gradients do not underflow. Returns a
 * new tensor, or the loss unchanged under bf16 (where scaling is unnecessary); NULL on error. */
Tensor* grad_scaler_scale(GradScaler* scaler, Tensor* loss) {
    if (!scaler || !loss) {
        LOG_ERROR("grad_scaler_scale: NULL argument");
        return NULL;
    }

    tensor_ensure_executed(loss);

    /* bf16 shares fp32's 8-bit exponent range, so gradients cannot underflow
     * the way they do in fp16 — loss scaling is unnecessary (a no-op). */
    if (autocast_get_dtype() == DTYPE_BFLOAT16)
        return loss;

    int scalar_shape[]  = {1};
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

/** Divide each parameter gradient in place by the scale factor, setting found_inf if any
 * inf/nan is seen so the following step can be skipped. */
void grad_scaler_unscale(GradScaler* scaler, Parameter** params, int num_params) {
    if (!scaler || !params) {
        LOG_ERROR("grad_scaler_unscale: NULL argument");
        return;
    }

    scaler->found_inf = false;
    /* Mirror grad_scaler_scale: under bf16 the loss was never scaled, so
     * unscale by 1.0 and keep only the inf/nan detection. */
    float inv_scale = (autocast_get_dtype() == DTYPE_BFLOAT16) ? 1.0f : 1.0f / scaler->scale_factor;

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

/** Invoke the optimizer step callback, but skip it when inf/nan gradients were detected. */
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

/** Adapt the scale after a step: back off (clamped to >= 1) on inf/nan, otherwise grow by
 * growth_factor once growth_interval clean steps elapse (clamped to 2^32). */
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
