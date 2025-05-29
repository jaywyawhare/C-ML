#ifndef SIGMOID_H
#define SIGMOID_H

// Forward declaration
typedef struct Node Node;

float sigmoid_scalar(float x);
Node *sigmoid_node(Node *x);
void sigmoid_backward(float grad_output, Node **inputs, int ninputs);

#endif
