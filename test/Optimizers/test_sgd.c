#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../include/Optimizers/sgd.h"
#include "../../include/Core/error_codes.h"

void test_sgd()
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

    float loss = sgd(x, y, lr, &w, &b);

    assert(fabs(loss - expected_loss) < loss_thresh);
    assert(fabs(w - w_expected) < loss_thresh);
    assert(fabs(b - b_expected) < loss_thresh);

    x = 3.0f;
    y = 7.0f;
    lr = 0.01f;

    loss = sgd(x, y, lr, NULL, NULL);
    assert(loss == CM_NULL_POINTER_ERROR);

    w = 2.0f;
    loss = sgd(x, y, lr, &w, NULL);
    assert(loss == CM_NULL_POINTER_ERROR);

    float b_val = 1.0f;
    loss = sgd(x, y, lr, NULL, &b_val);
    assert(loss == CM_NULL_POINTER_ERROR);

    float nan_val = NAN;
    float inf_val = INFINITY;

    w = 2.0f;
    b_val = 1.0f;

    loss = sgd(nan_val, y, lr, &w, &b_val);
    assert(isnan(loss));

    loss = sgd(inf_val, y, lr, &w, &b_val);
    assert(loss == CM_INVALID_INPUT_ERROR);

    loss = sgd(x, nan_val, lr, &w, &b_val);
    assert(isnan(loss));

    loss = sgd(x, inf_val, lr, &w, &b_val);
    assert(loss == CM_INVALID_INPUT_ERROR);

    float pred_thresh = 1e-1f;
    loss_thresh = 1e-6f;
    w = 5.0f;
    b_val = 3.0f;
    x = 3.0f;
    y = 7.0f;
    lr = 0.01f;
    float prev_loss = INFINITY;

    for (int i = 0; i < 100; i++)
    {
        loss = sgd(x, y, lr, &w, &b_val);
        assert(loss <= prev_loss || fabs(loss - prev_loss) < loss_thresh);
        prev_loss = loss;
    }

    y_pred = w * x + b_val;
    assert(fabs(y_pred - y) < pred_thresh);

    x = 3.0f;
    y = 7.0f;
    lr = -0.01f;
    w = 2.0f;
    b_val = 1.0f;

    sgd(x, y, lr, &w, &b_val);
}

int main()
{
    printf("Testing SGD\n");
    test_sgd();
    printf("SGD test passed\n");
    return 0;
}
