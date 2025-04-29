#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

#include "../../include/Core/autograd.h"

float leaky_relu(float x);
Node *leaky_relu_node(Node *x);
void leaky_relu_backward(float grad_output, Node **inputs, int ninputs);

#endif
