#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../include/Loss_Functions/focal_loss.h"
#include "../../include/Core/error_codes.h"

void test_focal_loss()
{
    float y[3] = {1.0, 0.0, 1.0};
    float yHat[3] = {0.9, 0.1, 0.8};
    float gamma = 2.0;

    float expected = (-1.0 * powf(1 - 0.9, 2.0) * logf(0.9) - (1 - 0.0) * powf(0.1, 2.0) * logf(0.9) - 1.0 * powf(1 - 0.8, 2.0) * logf(0.8)) / 3.0f;

    float tolerance = 1e-6;
    assert(fabs(focal_loss(y, yHat, 3, gamma) - expected) < tolerance);

    float y4[3] = {1.0, 0.0, 1.0};
    float yHat4[3] = {0.9, 0.1, 0.8};
    float gamma0 = 0.0;
    float expected0 = (-1.0 * logf(0.9) - (1 - 0.0) * logf(0.9) - 1.0 * logf(0.8)) / 3.0f;
    assert(fabs(focal_loss(y4, yHat4, 3, gamma0) - expected0) < tolerance);

    assert(focal_loss(NULL, yHat, 3, gamma) == CM_INVALID_INPUT_ERROR);
    assert(focal_loss(y, NULL, 3, gamma) == CM_INVALID_INPUT_ERROR);

    assert(focal_loss(y, yHat, 0, gamma) == CM_INVALID_INPUT_ERROR);

    printf("focal_loss test passed\n");
}

int main()
{
    printf("Testing focal_loss\n");
    test_focal_loss();
    return 0;
}
