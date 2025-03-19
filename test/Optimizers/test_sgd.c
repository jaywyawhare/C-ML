#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../src/my_functions.h"

void test_sgd_normal_case()
{
    float loss_thresh = 1e-6f;
    float w = 2.0f;
    float b = 1.0f;
    float x = 3.0f;
    float y = 7.0f;
    float lr = 0.01f;

    float y_pred = w * x + b;
    float expected_loss = pow(y_pred - y, 2);
    float dw_expected = 2 * (y_pred - y) * x;
    float db_expected = 2 * (y_pred - y);
    float w_expected = w - lr * dw_expected;
    float b_expected = b - lr * db_expected;

    float loss = SGD(x, y, lr, &w, &b);

    assert(fabs(loss - expected_loss) < loss_thresh);
    assert(fabs(w - w_expected) < loss_thresh);
    assert(fabs(b - b_expected) < loss_thresh);
}

void test_sgd_error_case()
{
    float error_thresh = 1e-6f;
    float x = 3.0f;
    float y = 7.0f;
    float lr = 0.01f;

    float loss = SGD(x, y, lr, NULL, NULL);
    assert(loss == -1);

    float w = 2.0f;
    loss = SGD(x, y, lr, &w, NULL);
    assert(loss == -1);

    float b = 1.0f;
    loss = SGD(x, y, lr, NULL, &b);
    assert(loss == -1);
}

void test_sgd_multiple_iterations()
{
    float pred_thresh = 1e-1f;
    float loss_thresh = 1e-6f;
    float w = 5.0f;
    float b = 3.0f;
    float x = 3.0f;
    float y = 7.0f;
    float lr = 0.01f;
    float prev_loss = INFINITY;

    for (int i = 0; i < 100; i++)
    {
        float loss = SGD(x, y, lr, &w, &b);
        assert(loss <= prev_loss || fabs(loss - prev_loss) < loss_thresh);
        prev_loss = loss;
    }

    float y_pred = w * x + b;
    assert(fabs(y_pred - y) < pred_thresh);
}

int main()
{
    printf("Testing SGD\n");
    test_sgd_normal_case();
    test_sgd_error_case();
    test_sgd_multiple_iterations();
    printf("SGD test passed\n");
    return 0;
}
