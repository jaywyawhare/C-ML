#ifndef RELU_H
#define RELU_H

// Forward declaration
typedef struct Node Node;

float relu_scalar(float x);
Node *relu_node(Node *x);
void relu_backward(float grad_output, Node **inputs, int ninputs);

#endif
