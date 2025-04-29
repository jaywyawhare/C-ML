#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../../include/Core/autograd.h"

float *softmax(float *z, int n);
Node *softmax_node(Node *x);
void softmax_backward(float grad_output, Node **inputs, int ninputs); // Add backward function

#endif
