#ifndef TANH_H
#define TANH_H
#include "../../include/Core/autograd.h"

float tanH(float x);
Node *tanh_node(Node *x);
void tanh_backward(float grad_output, Node **inputs, int ninputs); // Add backward declaration

#endif
