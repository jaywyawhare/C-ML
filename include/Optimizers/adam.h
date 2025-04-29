#ifndef ADAM_H
#define ADAM_H

#include <stdbool.h>
#include "regularization.h"

typedef struct
{
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float max_grad_norm;
    float clip_value;
    bool amsgrad;
    bool maximize;
    RegularizerConfig regularizer;
    LRSchedulerType lr_scheduler;
    float lr_gamma;
    int lr_step_size;
} AdamConfig;

float adam(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b,
           float *s_w, float *s_b, float *max_s_w, float *max_s_b, AdamConfig config,
           int epoch);

#endif
