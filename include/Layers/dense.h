#ifndef DENSE_H
#define DENSE_H

#include "../Core/autograd.h"
#include "../Optimizers/optimizer_types.h"
#include "../Optimizers/adam.h"
#include "../Optimizers/rmsprop.h"
#include "../Optimizers/sgd.h"
#include <stdlib.h>

typedef struct
{
    float *weights;
    float *biases;
    int input_size;
    int output_size;

    OptimizerType optimizer_type;
    float learning_rate;
    int step;

    AdamConfig adam_config;
    float *adam_m_w; 
    float *adam_m_b; 
    float *adam_v_w; 
    float *adam_v_b; 

    RMSpropConfig rmsprop_config;
    float *rms_cache_w;
    float *rms_cache_b;

    SGDConfig sgd_config;
} DenseLayer;

int initialize_dense(DenseLayer *layer, int input_size, int output_size);
int forward_dense(DenseLayer *layer, float *input, float *output);
int backward_dense(DenseLayer *layer, float *input, float *output, float *d_output, float *d_input);
int free_dense(DenseLayer *layer);

#endif
