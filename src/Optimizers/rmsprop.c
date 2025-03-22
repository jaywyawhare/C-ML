#include <math.h>
#include "../../include/Optimizers/rmsprop.h"

float RMSprop(float x, float y, float lr, float *w, float *b, float *cache_w, float *cache_b, float epsilon, float beta)
{

    if (!w || !b || !cache_w || !cache_b)
    {
        return -1;
    }

    if (epsilon <= 0)
    {
        return -1;
    }

    float y_pred = (*w) * x + (*b);
    float loss = pow(y_pred - y, 2);
    float dw = 2 * (y_pred - y) * x;
    float db = 2 * (y_pred - y);

    *cache_w = beta * (*cache_w) + (1 - beta) * (dw * dw);
    *cache_b = beta * (*cache_b) + (1 - beta) * (db * db);

    *w -= lr * (dw / (sqrt(*cache_w) + epsilon));
    *b -= lr * (db / (sqrt(*cache_b) + epsilon));

    return loss;
}
