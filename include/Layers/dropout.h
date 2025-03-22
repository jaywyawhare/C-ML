#ifndef DROPOUT_H
#define DROPOUT_H

typedef struct
{
    float dropout_rate;
} DropoutLayer;

void initializeDropout(DropoutLayer *layer, float dropout_rate);
void forwardDropout(DropoutLayer *layer, float *input, float *output, int size);
void backwardDropout(DropoutLayer *layer, float *input, float *output, float *d_output, float *d_input, int size);

#endif
