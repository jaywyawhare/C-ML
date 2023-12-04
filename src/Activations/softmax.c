#include <math.h>
#include <stdlib.h>

float* softmax(float *z, int n) {
    float sum = 0;
    float *output = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        output[i] = expf(z[i]);
        sum += output[i];
    }

    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }

    return output; 
}
