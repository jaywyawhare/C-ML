#include<math.h>

float meanAbsoluteError(float *y, float *yHat, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += fabsf(y[i] - yHat[i]);
    }
    return sum / n;
}
