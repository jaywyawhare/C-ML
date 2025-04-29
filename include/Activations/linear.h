#ifndef LINEAR_H
#define LINEAR_H

#include "../../include/Core/autograd.h"

float linear(float x);
Node *linear_node(Node *x);
void linear_backward(float grad_output, Node **inputs, int ninputs);

#endif
