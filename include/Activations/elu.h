#ifndef ELU_H
#define ELU_H

#include "../../include/Core/autograd.h"

float elu(float x, float alpha);
Node *elu_node(Node *x, float alpha);
void elu_backward(float grad_output, Node **inputs, int ninputs);

#endif
