#ifndef C_ML_GELU_H
#define C_ML_GELU_H

// Forward declaration for Node type
typedef struct Node Node;

float gelu_scalar(float x);
Node *gelu_node(Node *x);
void gelu_backward(float grad_output, Node **inputs, int ninputs);

#endif
