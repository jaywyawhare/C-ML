#include "../../include/Loss_Functions/reduce_mean.h"

float reduceMean(float *loss, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += loss[i];
    }
    if (size == 0)
    {
        return 0;
    }
    return sum / size;
}