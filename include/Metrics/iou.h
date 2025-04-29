#ifndef IOU_H
#define IOU_H

#include "../Core/autograd.h"

Node *iou(Node *y, Node *yHat, int n, float threshold);

#endif
