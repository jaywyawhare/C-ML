#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../src/my_functions.h"

void test_rmsprop_normal_case()
{
    float loss_thresh = 1e-6f;
    float w = 2.0f;
    float b = 1.0f;
    float x = 3.0f;
    float y = 7.0f;
    float lr = 0.01f;
    float cache_w = 0.0f;
    float cache_b = 0.0f;
    float eps = 1e-8f;
    float beta = 0.9f;

    float y_pred = w * x + b;
    float expected_loss = pow(y_pred - y, 2);
    float loss = RMSprop(x, y, lr, &w, &b, &cache_w, &cache_b, eps, beta);

    assert(fabs(loss - expected_loss) < loss_thresh);
}

void test_rmsprop_error_cases()
{
    float error_thresh = 1e-6f;
    float w = 2.0f;
    float b = 1.0f;
    float x = 3.0f;
    float y = 7.0f;
    float lr = 0.01f;
    float cache_w = 0.0f;
    float cache_b = 0.0f;
    float eps_invalid = 0.0f;
    float beta = 0.9f;
    float loss;

    loss = RMSprop(x, y, lr, &w, &b, &cache_w, &cache_b, eps_invalid, beta);
    assert(loss == -1);

    eps_invalid = 1e-8f;
    loss = RMSprop(x, y, lr, NULL, &b, &cache_w, &cache_b, eps_invalid, beta);
    assert(loss == -1);

    loss = RMSprop(x, y, lr, &w, NULL, &cache_w, &cache_b, eps_invalid, beta);
    assert(loss == -1);

    loss = RMSprop(x, y, lr, &w, &b, NULL, &cache_b, eps_invalid, beta);
    assert(loss == -1);

    loss = RMSprop(x, y, lr, &w, &b, &cache_w, NULL, eps_invalid, beta);
    assert(loss == -1);
}

void test_rmsprop_multiple_iterations()
{
    float final_pred_thresh = 1e-3f;
    float loss_thresh = 1e-6f;
    float w = 5.0f;
    float b = 3.0f;
    float x = 3.0f;
    float y = 7.0f;
    float lr = 0.01f;
    float cache_w = 0.0f;
    float cache_b = 0.0f;
    float eps = 1e-8f;
    float beta = 0.9f;
    float prev_loss = INFINITY;
    int iterations = 3000;
    int i;

    for (i = 0; i < iterations; i++)
    {
        float loss = RMSprop(x, y, lr, &w, &b, &cache_w, &cache_b, eps, beta);
        assert(loss <= prev_loss || fabs(loss - prev_loss) < loss_thresh);
        prev_loss = loss;
    }

    float y_pred_final = w * x + b;
    assert(fabs(y_pred_final - y) < final_pred_thresh);
}

int main()
{
    printf("Testing RMSprop\n");
    test_rmsprop_normal_case();
    test_rmsprop_error_cases();
    test_rmsprop_multiple_iterations();
    printf("RMSprop test passed\n");
    return 0;
}
