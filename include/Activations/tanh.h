#ifndef TANH_H
#define TANH_H

// Forward declaration
typedef struct Node Node;

float tanH_scalar(float x);
void tanh_backward(float grad_output, Node **inputs, int ninputs); // Add backward declaration

#endif
