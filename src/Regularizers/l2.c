#include<math.h>

float l2(float x, float y, float lr, float w, float b, float v_w, float v_b, float s_w, float s_b, float beta1, float beta2, float epsilon) {
    float y_pred = w * x + b;
    float loss = pow(y_pred - y, 2);
    float dw = 2 * (y_pred - y) * x;
    float db = 2 * (y_pred - y);
    v_w = beta1 * v_w + (1 - beta1) * dw;
    v_b = beta1 * v_b + (1 - beta1) * db;
    s_w = beta2 * s_w + (1 - beta2) * pow(dw, 2);
    s_b = beta2 * s_b + (1 - beta2) * pow(db, 2);
    w -= lr * v_w / (sqrt(s_w) + epsilon);
    b -= lr * v_b / (sqrt(s_b) + epsilon);
    return loss;
}
