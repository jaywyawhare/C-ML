#ifndef COHENS_KAPPA_H
#define COHENS_KAPPA_H

#include "../Core/autograd.h"

Node* cohens_kappa(Node *y, Node *yHat, int n, float threshold);

#endif
