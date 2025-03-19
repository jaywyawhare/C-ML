#include <math.h>

float meanAbsolutePercentageError(float *y, float *yHat, int n)
{
    float sum = 0;
    int valid_count = 0;
    for (int i = 0; i < n; i++)
    {
        if (fabsf(y[i]) < 1e-15)
        {
            continue;
        }
        sum += fabsf((y[i] - yHat[i]) / y[i]);
        valid_count++;
    }
    if (valid_count == 0)
    {
        return 0;
    }
    return sum / valid_count * 100;
}
