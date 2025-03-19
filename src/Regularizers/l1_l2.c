#include <math.h>

float l1_l2(float *w, float *dw, float *w_l1, float *dw_l1, float *w_l2, float *dw_l2, float l1, float l2, int n)
{
    float loss = 0;
    for (int i = 0; i < n; i++)
    {
        loss += l1 * fabs(w[i]) + l2 * pow(w[i], 2);
        dw[i] += l1 * (w[i] > 0 ? 1 : -1) + 2 * l2 * w[i];
        dw_l1[i] += (w[i] > 0 ? 1 : -1);
        dw_l2[i] += 2 * w[i];
    }
    return loss;
}
