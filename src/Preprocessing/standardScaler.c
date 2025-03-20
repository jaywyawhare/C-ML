#include <math.h>
#include <stdlib.h>
#include <stdio.h>

float *standardScaler(float *x, int size)
{
    if (size == 0) 
    {
        fprintf(stderr, "Input size is zero\n");
        return NULL;
    }

    float *scaled = malloc(sizeof(float) * size);
    if (scaled == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    float mean = 0;
    for (int i = 0; i < size; i++)
    {
        mean += x[i];
    }
    mean /= size;

    float std = 0;
    for (int i = 0; i < size; i++)
    {
        std += pow(x[i] - mean, 2);
    }
    std /= size;
    std = sqrt(std);

    if (std == 0)
    {
        fprintf(stderr, "Standard deviation is zero\n");
        free(scaled);
        return NULL;
    }

    for (int i = 0; i < size; i++)
    {
        scaled[i] = (x[i] - mean) / std;
    }
    return scaled;
}