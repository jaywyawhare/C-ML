#ifndef MEAN_ABSOLUTE_ERROR_H
#define MEAN_ABSOLUTE_ERROR_H

#include "../Core/autograd.h"

Node *mean_absolute_error(Node *y, Node *yHat, int n);

#endif
