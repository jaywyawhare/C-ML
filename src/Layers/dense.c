#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct
{
    float *weights;
    float *biases;
    int input_size;
    int output_size;
} DenseLayer;

void initializeDense(DenseLayer *layer, int input_size, int output_size)
{
    if (layer == NULL)
    {
        fprintf(stderr, "Layer is NULL\n");
        exit(1);
    }

    if (layer->weights != NULL)
    {
        free(layer->weights);
    }
    if (layer->biases != NULL)
    {
        free(layer->biases);
    }

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = (float *)malloc(input_size * output_size * sizeof(float));
    layer->biases = (float *)malloc(output_size * sizeof(float));

    if (layer->weights == NULL || layer->biases == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < input_size * output_size; i++)
    {
        layer->weights[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < output_size; i++)
    {
        layer->biases[i] = ((float)rand() / RAND_MAX) - 0.5;
    }
}

void forwardDense(DenseLayer *layer, float *input, float *output)
{
    if (layer == NULL || input == NULL || output == NULL)
    {
        fprintf(stderr, "Layer, input, or output is NULL\n");
        exit(1);
    }

    for (int i = 0; i < layer->output_size; i++)
    {
        output[i] = 0;
        for (int j = 0; j < layer->input_size; j++)
        {
            output[i] += input[j] * layer->weights[j + i * layer->input_size];
        }
        output[i] += layer->biases[i];
    }
}

void backwardDense(DenseLayer *layer, float *input, float *output, float *d_output, float *d_input, float *d_weights, float *d_biases)
{
    if (layer == NULL || input == NULL || output == NULL || d_output == NULL || d_input == NULL || d_weights == NULL || d_biases == NULL)
    {
        fprintf(stderr, "One or more arguments are NULL\n");
        exit(1);
    }

    for (int i = 0; i < layer->input_size; i++)
    {
        d_input[i] = 0;
        for (int j = 0; j < layer->output_size; j++)
        {
            d_input[i] += d_output[j] * layer->weights[i + j * layer->input_size];
        }
    }

    for (int i = 0; i < layer->output_size; i++)
    {
        d_biases[i] = d_output[i];
        for (int j = 0; j < layer->input_size; j++)
        {
            d_weights[j + i * layer->input_size] = input[j] * d_output[i];
        }
    }
}

void updateDense(DenseLayer *layer, float *d_weights, float *d_biases, float learning_rate)
{
    if (layer == NULL || d_weights == NULL || d_biases == NULL)
    {
        fprintf(stderr, "Layer or gradients are NULL\n");
        exit(1);
    }

    for (int i = 0; i < layer->input_size * layer->output_size; i++)
    {
        layer->weights[i] -= learning_rate * d_weights[i];
    }

    for (int i = 0; i < layer->output_size; i++)
    {
        layer->biases[i] -= learning_rate * d_biases[i];
    }
}

void freeDense(DenseLayer *layer)
{
    if (layer == NULL)
    {
        return;
    }

    if (layer->weights != NULL)
    {
        free(layer->weights);
        layer->weights = NULL; 
    }
    if (layer->biases != NULL)
    {
        free(layer->biases);
        layer->biases = NULL; 
    }
}