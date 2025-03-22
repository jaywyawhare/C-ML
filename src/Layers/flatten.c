#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../../include/Layers/flatten.h"


int initializeFlatten(FlattenLayer *layer, int input_size)
{
    if (layer == NULL)
    {
        fprintf(stderr, "Layer is NULL\n");
        return -1;
    }

    if (input_size <= 0)
    {
        fprintf(stderr, "Invalid input size\n");
        return -1; 
    }

    layer->input_size = input_size;
    layer->output_size = input_size;
    return 0; 
}

void forwardFlatten(FlattenLayer *layer, float *input, float *output)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        fprintf(stderr, "Layer, input, or output is NULL\n");
        exit(1);
    }

    if (layer->input_size <= 0)
    {
        fprintf(stderr, "Invalid input size\n");
        exit(1);
    }

    for (int i = 0; i < layer->input_size; i++)
    {
        output[i] = input[i];
    }
}

void backwardFlatten(FlattenLayer *layer, float *input, float *output, float *d_output, float *d_input)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL)
    {
        fprintf(stderr, "One or more arguments are NULL\n");
        exit(1);
    }

    if (layer->input_size <= 0)
    {
        fprintf(stderr, "Invalid input size\n");
        exit(1);
    }

    for (int i = 0; i < layer->input_size; i++)
    {
        d_input[i] = d_output[i];
    }
}

void freeFlatten(FlattenLayer *layer)
{
    if (layer == NULL)
    {
        fprintf(stderr, "Layer is NULL\n");
        exit(1);
    }
}
