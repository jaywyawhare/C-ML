#include <math.h>

float RMSprop(float x, float y, float lr, float *w, float *b, float *cache_w, float *cache_b, float epsilon, float beta1, float beta2) {
    float y_pred = (*w) * x + (*b);
    float loss = pow(y_pred - y, 2);
    float dw = 2 * (y_pred - y) * x;
    float db = 2 * (y_pred - y);

    *cache_w = beta1 * (*cache_w) + (1 - beta1) * (dw * dw);
    *cache_b = beta1 * (*cache_b) + (1 - beta1) * (db * db);

    float cache_w_corrected = *cache_w / (1 - pow(beta1, 2));
    float cache_b_corrected = *cache_b / (1 - pow(beta1, 2));

    *w -= lr * (dw / (sqrt(cache_w_corrected) + epsilon));
    *b -= lr * (db / (sqrt(cache_b_corrected) + epsilon));

    return loss;
} 
