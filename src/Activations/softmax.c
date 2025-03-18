#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float* softmax(float *z, int n) {
    float max_val = z[0];
    for (int i = 1; i < n; i++) {
        if (z[i] > max_val) {
            max_val = z[i];
        }
    }

    float sum = 0;
    float *output = (float *)malloc(n * sizeof(float));

    if (output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        output[i] = expf(z[i] - max_val); 
        sum += output[i];
    }

    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }

    return output;
}

void freeSoftmax(float *output) {
    free(output);
}