#include <math.h>

float focalLoss(float *y, float *yHat, int n, float gamma)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += -y[i] * powf(1 - yHat[i], gamma) * logf(fmaxf(yHat[i], 1e-15)) - (1 - y[i]) * powf(yHat[i], gamma) * logf(fmaxf(1 - yHat[i], 1e-15));
    }
    return sum / n;
}