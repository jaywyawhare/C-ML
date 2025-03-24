#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../include/Optimizers/rmsprop.h"
#include "../../include/Core/error_codes.h"

void test_rmsprop()
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
    float loss = rms_prop(x, y, lr, &w, &b, &cache_w, &cache_b, eps, beta);

    assert(fabs(loss - expected_loss) < loss_thresh);

    w = 2.0f;
    b = 1.0f;
    x = 3.0f;
    y = 7.0f;
    lr = 0.01f;
    cache_w = 0.0f;
    cache_b = 0.0f;
    float eps_invalid = 0.0f;
    beta = 0.9f;

    loss = rms_prop(x, y, lr, &w, &b, &cache_w, &cache_b, eps_invalid, beta);
    assert(loss == CM_INVALID_INPUT_ERROR);

    eps_invalid = 1e-8f;
    loss = rms_prop(x, y, lr, NULL, &b, &cache_w, &cache_b, eps_invalid, beta);
    assert(loss == CM_NULL_POINTER_ERROR);

    loss = rms_prop(x, y, lr, &w, NULL, &cache_w, &cache_b, eps_invalid, beta);
    assert(loss == CM_NULL_POINTER_ERROR);

    loss = rms_prop(x, y, lr, &w, &b, NULL, &cache_b, eps_invalid, beta);
    assert(loss == CM_NULL_POINTER_ERROR);

    loss = rms_prop(x, y, lr, &w, &b, &cache_w, NULL, eps_invalid, beta);
    assert(loss == CM_NULL_POINTER_ERROR);

    float nan_val = NAN;
    float inf_val = INFINITY;

    loss = rms_prop(nan_val, y, lr, &w, &b, &cache_w, &cache_b, eps_invalid, beta);
    assert(isnan(loss));

    loss = rms_prop(inf_val, y, lr, &w, &b, &cache_w, &cache_b, eps_invalid, beta);
    assert(loss == CM_INVALID_INPUT_ERROR);

    loss = rms_prop(x, nan_val, lr, &w, &b, &cache_w, &cache_b, eps_invalid, beta);
    assert(isnan(loss));

    loss = rms_prop(x, inf_val, lr, &w, &b, &cache_w, &cache_b, eps_invalid, beta);
    assert(loss == CM_INVALID_INPUT_ERROR);

    float final_pred_thresh = 1e-3f;
    loss_thresh = 1e-6f;
    w = 5.0f;
    b = 3.0f;
    x = 3.0f;
    y = 7.0f;
    lr = 0.01f;
    cache_w = 0.0f;
    cache_b = 0.0f;
    eps = 1e-8f;
    beta = 0.9f;
    float prev_loss = INFINITY;
    int iterations = 3000;
    int i;

    for (i = 0; i < iterations; i++)
    {
        loss = rms_prop(x, y, lr, &w, &b, &cache_w, &cache_b, eps, beta);
        assert(loss <= prev_loss || fabs(loss - prev_loss) < loss_thresh);
        prev_loss = loss;
    }

    float y_pred_final = w * x + b;
    assert(fabs(y_pred_final - y) < final_pred_thresh);

    w = 2.0f;
    b = 1.0f;
    x = 3.0f;
    y = 7.0f;
    lr = 0.01f;
    cache_w = 0.0f;
    cache_b = 0.0f;
    eps = 1e-8f;
    float beta_invalid = 1.1f;

    loss = rms_prop(x, y, lr, &w, &b, &cache_w, &cache_b, eps, beta_invalid);
    assert(isnan(loss) || !isnan(loss));
}

int main()
{
    printf("Testing RMSprop\n");
    test_rmsprop();
    printf("RMSprop test passed\n");
    return CM_SUCCESS;
}
