#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../src/Loss_Functions/focalLoss.h"

void test_focalLoss()
{
    float y[3] = {1.0, 0.0, 1.0};
    float yHat[3] = {0.9, 0.1, 0.8};
    float gamma = 2.0;

    float expected = (-1.0 * powf(1 - 0.9, 2.0) * logf(0.9) - (1 - 0.0) * powf(0.1, 2.0) * logf(0.9) - 1.0 * powf(1 - 0.8, 2.0) * logf(0.8)) / 3.0f;

    float tolerance = 1e-6;
    assert(fabs(focalLoss(y, yHat, 3, gamma) - expected) < tolerance);

    printf("focalLoss test passed\n");
}

int main()
{
    printf("Testing focalLoss\n");
    test_focalLoss();
    return 0;
}
