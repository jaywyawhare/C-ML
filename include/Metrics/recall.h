#ifndef RECALL_H
#define RECALL_H

#include "../Core/autograd.h"

Node* recall(Node *y, Node *yHat, int n, float threshold);

#endif
