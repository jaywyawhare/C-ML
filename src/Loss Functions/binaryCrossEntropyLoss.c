#include <math.h>

float binaryCrossEntropyLoss(float *yHat, float *y, int size)
{
    float loss = 0.0;

    for (int i = 0; i < size; ++i)
    {

        float epsilon = 1e-15;
        float predicted = fmax(fmin(yHat[i], 1 - epsilon), epsilon);
        loss += -(y[i] * log(predicted) + (1 - y[i]) * log(1 - predicted));
    }

    loss /= size;
    return loss;
}
