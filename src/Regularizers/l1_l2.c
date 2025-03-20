#include <math.h>
#include <stdlib.h>
#include <stdio.h>

float l1_l2(float *w, float *dw, float l1, float l2, int n)
{
    if (w == NULL || dw == NULL || n <= 0)
    {
        fprintf(stderr, "Error: Invalid input to l1_l2\n");
        return -1;
    }

    float loss = 0;
    for (int i = 0; i < n; i++)
    {
        loss += l1 * fabs(w[i]) + l2 * pow(w[i], 2);

        float l1_grad = (w[i] > 0) ? 1 : (w[i] < 0) ? -1
                                                    : 0;

        dw[i] += l1 * l1_grad + 2 * l2 * w[i];
    }
    return loss;
}