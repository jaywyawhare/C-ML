#ifndef ELU_H
#define ELU_H

#include "../../include/Core/autograd.h"

float elu_scalar(float x, float alpha);
Node *elu_node(Node *x, float alpha);
void elu_backward(Node *grad_output, Node *node);

#endif
