#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

// Forward declaration
typedef struct Node Node;

float leaky_relu_scalar(float x);
void leaky_relu_backward(float grad_output, Node **inputs, int ninputs);

#endif
