#include <math.h>

float binaryCrossEntropyLoss(float *yHat, float *y, int size) {
    float eps = 1e-15;
    float loss = 0;

    for (int i = 0; i < size; i++) {
        int true_class = y[i];
        float predicted_probability = yHat[i];
        if (true_class == 1) {
            loss += -logf(fmaxf(predicted_probability, eps));
        } else {
            loss += -logf(fmaxf(1 - predicted_probability, eps));
        }
    }
    return loss / size;
}
