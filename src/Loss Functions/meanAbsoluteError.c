#include <math.h>

float meanAbsoluteError(float *y, float *yHat, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += fabsf(y[i] - yHat[i]);
    }
    if (n == 0)
    {
        return 0;
    }
    return sum / n;
}
