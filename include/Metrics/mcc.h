#ifndef MCC_H
#define MCC_H

#include "../Core/autograd.h"

Node* mcc(Node *y, Node *yHat, int n, float threshold);

#endif
