#ifndef RELU_H
#define RELU_H
#include "../../include/Core/autograd.h"

float relu(float x);
Node *relu_node(Node *x);
void relu_backward(float grad_output, Node **inputs, int ninputs); // Add backward declaration

#endif
