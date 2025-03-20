#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../src/Optimizers/Adam.h"

void test_AdamOptimizer()
{
    float loss_thresh = 1e-6f;
    float param_thresh = 1e-4f;
    float final_param_thresh = 0.1f;

    float x = 1.0f;
    float y = 2.0f;
    float lr = 0.01f;
    float w = 0.5f;
    float b = 0.5f;
    float v_w = 0.0f;
    float v_b = 0.0f;
    float s_w = 0.0f;
    float s_b = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    float expected_loss = 1.0f;
    float expected_w = 0.51f;
    float expected_b = 0.51f;

    float actual_loss = Adam(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon);

    assert(fabs(actual_loss - expected_loss) < loss_thresh);
    assert(fabs(w - expected_w) < param_thresh);
    assert(fabs(b - expected_b) < param_thresh);

    float x_vals[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y_vals[5] = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};

    float w_train = 0.5f;
    float b_train = 0.5f;
    v_w = 0.0f;
    v_b = 0.0f;
    s_w = 0.0f;
    s_b = 0.0f;

    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < 5; i++)
        {
            Adam(x_vals[i], y_vals[i], lr, &w_train, &b_train, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon);
        }
    }

    assert(fabs(w_train - 2.0f) < final_param_thresh);
    assert(fabs(b_train - 1.0f) < final_param_thresh);
}

int main()
{
    printf("Testing AdamOptimizer\n");
    test_AdamOptimizer();
    printf("AdamOptimizer test passed\n");
    return 0;
}
