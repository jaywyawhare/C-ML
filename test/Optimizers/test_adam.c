#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../include/Optimizers/adam.h"
#include "../../include/Core/error_codes.h"

void test_adam()
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

    float actual_loss = adam(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon);

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
            adam(x_vals[i], y_vals[i], lr, &w_train, &b_train, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon);
        }
    }

    assert(fabs(w_train - 2.0f) < final_param_thresh);
    assert(fabs(b_train - 1.0f) < final_param_thresh);

    x = 1.0f;
    y = 2.0f;
    lr = 0.01f;
    w = 0.5f;
    b = 0.5f;
    v_w = 0.0f;
    v_b = 0.0f;
    s_w = 0.0f;
    s_b = 0.0f;
    beta1 = 0.9f;
    beta2 = 0.999f;
    epsilon = 1e-8f;
    assert(adam(x, y, lr, NULL, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon) == CM_NULL_POINTER_ERROR);
    assert(adam(x, y, lr, &w, NULL, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon) == CM_NULL_POINTER_ERROR);
    assert(adam(x, y, lr, &w, &b, NULL, &v_b, &s_w, &s_b, beta1, beta2, epsilon) == CM_NULL_POINTER_ERROR);
    assert(adam(x, y, lr, &w, &b, &v_w, NULL, &s_w, &s_b, beta1, beta2, epsilon) == CM_NULL_POINTER_ERROR);
    assert(adam(x, y, lr, &w, &b, &v_w, &v_b, NULL, &s_b, beta1, beta2, epsilon) == CM_NULL_POINTER_ERROR);
    assert(adam(x, y, lr, &w, &b, &v_w, &v_b, &s_w, NULL, beta1, beta2, epsilon) == CM_NULL_POINTER_ERROR);

    assert(adam(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, 0.0f) == CM_INVALID_INPUT_ERROR);
    assert(adam(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, -1.0f) == CM_INVALID_INPUT_ERROR);
    assert(adam(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, 1.0f, beta2, epsilon) == CM_INVALID_INPUT_ERROR);
    assert(adam(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, 1.0f, epsilon) == CM_INVALID_INPUT_ERROR);
    assert(adam(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, -0.1f, beta2, epsilon) == CM_INVALID_INPUT_ERROR);
    assert(adam(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, -0.1f, epsilon) == CM_INVALID_INPUT_ERROR);
    assert(adam(x, y, -0.01f, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon) == CM_INVALID_INPUT_ERROR);

    float nan_val = NAN;
    float inf_val = INFINITY;

    assert(adam(nan_val, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon) == CM_INVALID_INPUT_ERROR);
    assert(adam(inf_val, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon) == CM_INVALID_INPUT_ERROR);
    assert(adam(x, nan_val, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon) == CM_INVALID_INPUT_ERROR);
    assert(adam(x, inf_val, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon) == CM_INVALID_INPUT_ERROR);
}

int main()
{
    printf("Testing adam\n");
    test_adam();
    printf("adam test passed\n");
    return 0;
}
