#ifndef DENSE_H
#define DENSE_H

#include <stdio.h>

typedef struct
{
    float *weights;
    float *biases;
    int input_size;
    int output_size;
} DenseLayer;

void initializeDense(DenseLayer *layer, int input_size, int output_size);
void forwardDense(DenseLayer *layer, float *input, float *output);
void backwardDense(DenseLayer *layer, float *input, float *output, float *d_output, float *d_input, float *d_weights, float *d_biases);
void updateDense(DenseLayer *layer, float *d_weights, float *d_biases, float learning_rate);
void freeDense(DenseLayer *layer);

#endif
