#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../include/Layers/dropout.h"


void initializeDropout(DropoutLayer *layer, float dropout_rate)
{
    if (layer == NULL)
    {
        fprintf(stderr, "Layer is NULL\n");
        exit(1);
    }

    layer->dropout_rate = dropout_rate;
    srand((unsigned int)time(NULL));
}

void forwardDropout(DropoutLayer *layer, float *input, float *output, int size)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        fprintf(stderr, "Layer, input, or output is NULL\n");
        exit(1);
    }

    for (int i = 0; i < size; i++)
    {
        if ((float)rand() / RAND_MAX < layer->dropout_rate)
        {
            output[i] = 0;
        }
        else
        {
            output[i] = input[i] / (1 - layer->dropout_rate);
        }
    }
}

void backwardDropout(DropoutLayer *layer, float *input, float *output, float *d_output, float *d_input, int size)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL)
    {
        fprintf(stderr, "One or more arguments are NULL\n");
        exit(1);
    }

    for (int i = 0; i < size; i++)
    {
        if (output[i] == 0.0f)
        {
            d_input[i] = 0.0f;
        }
        else
        {
            d_input[i] = d_output[i] / (1 - layer->dropout_rate);
        }
    }
}
