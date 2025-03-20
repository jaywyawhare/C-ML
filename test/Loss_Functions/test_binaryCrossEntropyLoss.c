#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../src/Loss_Functions/binaryCrossEntropyLoss.h"

void test_binaryCrossEntropyLoss()
{
    float y[3] = {0.0, 1.0, 0.0};
    float yHat[3] = {0.1, 0.9, 0.2};
    float expected = 0.14462153f;
    float tolerance = 1e-6;

    float actual = binaryCrossEntropyLoss(yHat, y, 3);
    assert(fabs(actual - expected) < tolerance);

    float y2[3] = {1.0, 0.0, 1.0};
    float yHat2[3] = {0.9999, 0.0001, 0.9999};
    float expected2 = 0.0001f;

    float actual2 = binaryCrossEntropyLoss(yHat2, y2, 3);
    assert(fabs(actual2 - expected2) < tolerance);

    float y3[3] = {0.0, 1.0, 0.0};
    float yHat3[3] = {0.5, 0.5, 0.5};
    float expected3 = 0.693147181f;
    assert(fabs(binaryCrossEntropyLoss(yHat3, y3, 3) - expected3) < tolerance);

    float y_empty[0] = {};
    float yHat_empty[0] = {};
    assert(isnan(binaryCrossEntropyLoss(yHat_empty, y_empty, 0)) || binaryCrossEntropyLoss(yHat_empty, y_empty, 0) == 0);

    printf("binaryCrossEntropyLoss test passed\n");
}

int main()
{
    printf("Testing binaryCrossEntropyLoss\n");
    test_binaryCrossEntropyLoss();
    return 0;
}
