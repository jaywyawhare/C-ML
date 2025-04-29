#ifndef MEAN_ABSOLUTE_PERCENTAGE_ERROR_H
#define MEAN_ABSOLUTE_PERCENTAGE_ERROR_H

#include "../Core/autograd.h"

Node *mean_absolute_percentage_error(Node *y, Node *yHat, int n);

#endif
