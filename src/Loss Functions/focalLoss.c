#include<math.h>

float focalLoss(float *y, float *yHat, int n, float gamma) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += -y[i] * powf(1 - yHat[i], gamma) * logf(yHat[i]) - (1 - y[i]) * powf(yHat[i], gamma) * logf(1 - yHat[i]);
    }
    return sum / n;
}