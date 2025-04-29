#ifndef PRECISION_H
#define PRECISION_H

#include "../Core/autograd.h"

Node* precision(Node *y, Node *yHat, int n, float threshold);

#endif
