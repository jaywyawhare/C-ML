#ifndef FLATTEN_H
#define FLATTEN_H
#include <stdio.h>

typedef struct
{
    int input_size;
    int output_size;
} FlattenLayer;

int initialize_flatten(FlattenLayer *layer, int input_size);
int forward_flatten(FlattenLayer *layer, float *input, float *output);
int backward_flatten(FlattenLayer *layer, float *input, float *output, float *d_output, float *d_input);
int free_flatten(FlattenLayer *layer);

#endif