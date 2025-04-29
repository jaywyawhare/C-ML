#ifndef SGD_H
#define SGD_H

#include <stdbool.h>
#include "regularization.h"

typedef struct
{
    float momentum;
    float dampening;
    float weight_decay;
    float max_grad_norm;
    float clip_value;
    bool nesterov;
    bool maximize;
    RegularizerConfig regularizer;
    LRSchedulerType lr_scheduler;
    float lr_gamma;
    int lr_step_size;
} SGDConfig;

float sgd(float x, float y, float lr, float *w, float *b,
          float *v_w, float *v_b, SGDConfig config, int epoch);

#endif
