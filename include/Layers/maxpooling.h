#ifndef MAXPOOLING_H
#define MAXPOOLING_H
#include "../../include/Core/memory_management.h"

typedef struct
{
    int kernel_size;
    int stride;
} MaxPoolingLayer;

int initialize_maxpooling(MaxPoolingLayer *layer, int kernel_size, int stride);
int compute_maxpooling_output_size(int input_size, int kernel_size, int stride);
int forward_maxpooling(MaxPoolingLayer *layer, const float *input, float *output, int input_size);
int free_maxpooling(MaxPoolingLayer *layer);

#endif