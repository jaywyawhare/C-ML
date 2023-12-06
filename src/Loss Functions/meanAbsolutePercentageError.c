#include<math.h>

float meanAbsolutePercentageError(float *y, float *yHat, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += fabsf((y[i] - yHat[i]) / y[i]);
    }
    return sum / n * 100;
}
