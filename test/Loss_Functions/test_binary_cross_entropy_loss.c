#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../include/Loss_Functions/binary_cross_entropy_loss.h"
#include "../../include/Core/error_codes.h"

void test_binary_cross_entropy_loss()
{
    float y[3] = {0.0, 1.0, 0.0};
    float yHat[3] = {0.1, 0.9, 0.2};
    float expected = 0.14462153f;
    float tolerance = 1e-6;

    float actual = binary_cross_entropy_loss(yHat, y, 3);
    assert(fabs(actual - expected) < tolerance);

    float y2[3] = {1.0, 0.0, 1.0};
    float yHat2[3] = {0.9999, 0.0001, 0.9999};
    float expected2 = 0.0001f;

    float actual2 = binary_cross_entropy_loss(yHat2, y2, 3);
    assert(fabs(actual2 - expected2) < tolerance);

    float y3[3] = {0.0, 1.0, 0.0};
    float yHat3[3] = {0.5, 0.5, 0.5};
    float expected3 = 0.693147181f;
    assert(fabs(binary_cross_entropy_loss(yHat3, y3, 3) - expected3) < tolerance);

    float y_empty[0] = {};
    float yHat_empty[0] = {};
    float result = binary_cross_entropy_loss(yHat_empty, y_empty, 0);
    assert(result == CM_INVALID_INPUT_ERROR);

    assert(binary_cross_entropy_loss(NULL, y, 3) == CM_INVALID_INPUT_ERROR);
    assert(binary_cross_entropy_loss(yHat, NULL, 3) == CM_INVALID_INPUT_ERROR);

    assert(binary_cross_entropy_loss(yHat, y, 0) == CM_INVALID_INPUT_ERROR);

    printf("binary_cross_entropy_loss test passed\n");
}

int main()
{
    printf("Testing binary_cross_entropy_loss\n");
    test_binary_cross_entropy_loss();
    return 0;
}
