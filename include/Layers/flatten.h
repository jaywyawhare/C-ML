#ifndef FLATTEN_H
#define FLATTEN_H

#include <stdio.h>

typedef struct
{
    int input_size;
    int output_size;
} FlattenLayer;


int initializeFlatten(FlattenLayer *layer, int input_size);
void forwardFlatten(FlattenLayer *layer, float *input, float *output);
void backwardFlatten(FlattenLayer *layer, float *input, float *output, float *d_output, float *d_input);
void freeFlatten(FlattenLayer *layer);

#endif