#ifndef ROOT_MEAN_SQUARED_ERROR_H
#define ROOT_MEAN_SQUARED_ERROR_H

#include "../Core/autograd.h"

Node *root_mean_squared_error(Node *y, Node *yHat, int n);

#endif
