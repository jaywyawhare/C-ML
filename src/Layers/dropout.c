#include <stdlib.h>

typedef struct {
    float dropout_rate;
} DropoutLayer;

void initializeDropout(DropoutLayer *layer, float dropout_rate) {
    layer->dropout_rate = dropout_rate;
}

void forwardDropout(DropoutLayer *layer, float *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        if ((float)rand() / RAND_MAX < layer->dropout_rate) {
            output[i] = 0;
        } else {
            output[i] = input[i] / (1 - layer->dropout_rate);
        }
    }
}

void backwardDropout(DropoutLayer *layer, float *input, float *output, float *d_output, float *d_input, int size) {
    for (int i = 0; i < size; i++) {
        if (input[i] == 0) {
            d_input[i] = 0;
        } else {
            d_input[i] = d_output[i] / (1 - layer->dropout_rate);
        }
    }
}
