#include <math.h>

float sigmoid(float x)
{
    if (x >= 0)
    {
        float exp_neg_x = expf(-x);
        return 1 / (1 + exp_neg_x);
    }
    else
    {
        float exp_pos_x = expf(x);
        return exp_pos_x / (1 + exp_pos_x);
    }
}