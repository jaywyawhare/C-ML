#ifndef SIGMOID_H
#define SIGMOID_H
#include "../../include/Core/autograd.h"

float sigmoid(float x);
Node *sigmoid_node(Node *x);
void sigmoid_backward(float grad_output, Node **inputs, int ninputs); // Add backward declaration

#endif
