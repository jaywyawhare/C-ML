#ifndef C_ML_GELU_H
#define C_ML_GELU_H

#include "../../include/Core/autograd.h"

float gelu(float x);
Node *gelu_node(Node *x);
void gelu_backward(float grad_output, Node **inputs, int ninputs);

#endif
