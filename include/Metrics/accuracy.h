#ifndef ACCURACY_H
#define ACCURACY_H

#include "../Core/autograd.h"

Node *accuracy(Node *y, Node *yHat, int n, float threshold);

#endif
