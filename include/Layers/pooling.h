#ifndef POOLING_H
#define POOLING_H
#include "../../include/Core/memory_management.h"

typedef struct
{
    int kernel_size;
    int stride;
} PoolingLayer;

int initialize_pooling(PoolingLayer *layer, int kernel_size, int stride);
int compute_pooling_output_size(int input_size, int kernel_size, int stride);
int forward_pooling(PoolingLayer *layer, const float *input, float *output, int input_size);
int free_pooling(PoolingLayer *layer);

#endif