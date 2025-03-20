#include <math.h>
#include <stdlib.h>
#include <stdio.h>

float *minMaxScaler(float *x, int size)
{
    float *scaled = malloc(sizeof(float) * size);
    if (scaled == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    float min = x[0];
    float max = x[0];
    for (int i = 0; i < size; i++)
    {
        if (x[i] < min)
        {
            min = x[i];
        }
        if (x[i] > max)
        {
            max = x[i];
        }
    }
    if (max == min)
    {
        fprintf(stderr, "Max and min are equal\n");
        free(scaled);
        return NULL;
    }
    for (int i = 0; i < size; i++)
    {
        scaled[i] = (x[i] - min) / (max - min);
    }
    return scaled;
}