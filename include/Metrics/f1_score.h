#ifndef F1_SCORE_H
#define F1_SCORE_H

#include "../Core/autograd.h"

Node *f1_score(Node *y, Node *yHat, int n, float threshold);

#endif
