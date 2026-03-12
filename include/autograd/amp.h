#ifndef CML_AUTOGRAD_AMP_H
#define CML_AUTOGRAD_AMP_H

#include "tensor/tensor.h"
#include "autograd/autograd.h"
#include <stdbool.h>

struct Parameter;
typedef struct Parameter Parameter;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AutocastContext {
    bool enabled;
    DType target_dtype;   /* target low-precision dtype (e.g. DTYPE_FLOAT16) */
} AutocastContext;

/*
 * Gradient scaler for mixed-precision training.
 * Scales the loss to prevent gradient underflow in float16, then unscales
 * before the optimizer step. Dynamically adjusts based on inf/nan detection.
 */
typedef struct GradScaler {
    float scale_factor;
    float growth_factor;      /* default: 2.0 */
    float backoff_factor;     /* default: 0.5 */
    int growth_interval;      /* default: 2000 */
    int growth_step;
    bool found_inf;
} GradScaler;

void autocast_enter(DType target_dtype);
void autocast_exit(void);
bool autocast_is_enabled(void);
AutocastContext* autocast_get_context(void);

/* Ops like softmax, layer norm, losses should stay in float32 */
bool autocast_should_keep_float32(OpType op);

GradScaler* grad_scaler_create(float init_scale, float growth_factor,
                                float backoff_factor, int growth_interval);
void grad_scaler_free(GradScaler* scaler);
Tensor* grad_scaler_scale(GradScaler* scaler, Tensor* loss);
void grad_scaler_unscale(GradScaler* scaler, Parameter** params, int num_params);

/* Calls step_fn only if no inf/nan gradients were found */
void grad_scaler_step(GradScaler* scaler, void (*step_fn)(void*), void* optimizer);

void grad_scaler_update(GradScaler* scaler);

#ifdef __cplusplus
}
#endif

#endif // CML_AUTOGRAD_AMP_H
