#ifndef BALANCED_ACCURACY_H
#define BALANCED_ACCURACY_H

#include "../Core/autograd.h"

Node *balanced_accuracy(Node *y, Node *yHat, int n, float threshold);

#endif
