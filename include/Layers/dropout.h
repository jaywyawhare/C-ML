#ifndef DROPOUT_H
#define DROPOUT_H

typedef struct
{
    float dropout_rate;
} DropoutLayer;

int initialize_dropout(DropoutLayer *layer, float dropout_rate);
int forward_dropout(DropoutLayer *layer, float *input, float *output, int size);
int backward_dropout(DropoutLayer *layer, float *input, float *output, float *d_output, float *d_input, int size);

#endif
