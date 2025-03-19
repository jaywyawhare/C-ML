#include <math.h>

float Adam(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b, float *s_w, float *s_b, float beta1, float beta2, float epsilon)
{

    if (!w || !b || !v_w || !v_b || !s_w || !s_b)
    {
        return -1;
    }

    if (epsilon <= 0 || beta1 >= 1.0 || beta2 >= 1.0 || beta1 <= 0.0 || beta2 <= 0.0 || lr <= 0)
    {
        return -1;
    }

    static int t = 0;
    t++;

    float y_pred = *w * x + *b;
    float loss = pow(y_pred - y, 2);

    float dw = 2 * (y_pred - y) * x;
    float db = 2 * (y_pred - y);

    *v_w = beta1 * *v_w + (1 - beta1) * dw;
    *v_b = beta1 * *v_b + (1 - beta1) * db;
    *s_w = beta2 * *s_w + (1 - beta2) * dw * dw;
    *s_b = beta2 * *s_b + (1 - beta2) * db * db;

    float v_w_corrected = *v_w / (1 - pow(beta1, t));
    float v_b_corrected = *v_b / (1 - pow(beta1, t));
    float s_w_corrected = *s_w / (1 - pow(beta2, t));
    float s_b_corrected = *s_b / (1 - pow(beta2, t));

    *w -= lr * v_w_corrected / (sqrt(s_w_corrected + epsilon));
    *b -= lr * v_b_corrected / (sqrt(s_b_corrected + epsilon));

    return loss;
}
