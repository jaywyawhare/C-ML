#ifndef RMSPROP_H
#define RMSPROP_H

#include <stdbool.h>
#include "regularization.h"

typedef struct
{
    float alpha;
    float eps;
    float momentum;
    float weight_decay;
    float max_grad_norm;
    float clip_value;
    bool centered;
    RegularizerConfig regularizer;
    LRSchedulerType lr_scheduler;
    float lr_gamma;
    int lr_step_size;
} RMSpropConfig;

float rmsprop(float x, float y, float lr, float *w, float *b,
              float *cache_w, float *cache_b, float *v_w, float *v_b,
              float *avg_grad_w, float *avg_grad_b, RMSpropConfig config,
              int epoch);

#endif
