#include <math.h>

float SGD(float x, float y, float lr, float *w, float *b)
{
    if (!w || !b)
    {
        return -1;
    }

    float y_pred = (*w) * x + (*b);
    float loss = pow(y_pred - y, 2);
    float dw = 2 * (y_pred - y) * x;
    float db = 2 * (y_pred - y);

    (*w) -= lr * dw;
    (*b) -= lr * db;

    return loss;
}
