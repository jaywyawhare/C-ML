#ifndef SOFTMAX_H
#define SOFTMAX_H

// Forward declaration
typedef struct Node Node;

float *softmax_array(float *z, int n);
Node *softmax_node(Node *x);
void softmax_backward(float grad_output, Node **inputs, int ninputs);

#endif
