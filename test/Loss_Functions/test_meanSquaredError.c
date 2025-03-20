#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../src/Loss_Functions/meanSquaredError.h"

void test_meanSquaredError()
{
    float y[3] = {1.0, 2.0, 3.0};
    float yHat[3] = {1.1, 1.9, 3.2};
    float expected = 0.02f;
    float tolerance = 1e-6;
    assert(fabs(meanSquaredError(y, yHat, 3) - expected) < tolerance);

    float y_empty[0] = {};
    float yHat_empty[0] = {};
    assert(meanSquaredError(y_empty, yHat_empty, 0) == 0);

    float y_identical[3] = {1.0, 2.0, 3.0};
    float yHat_identical[3] = {1.0, 2.0, 3.0};
    assert(meanSquaredError(y_identical, yHat_identical, 3) == 0);

    float y_large[3] = {1e6, 2e6, 3e6};
    float yHat_large[3] = {1e6 + 1, 2e6 - 1, 3e6 + 2};
    float expected_large = (1 + 1 + 4) / 3.0f;
    assert(fabs(meanSquaredError(y_large, yHat_large, 3) - expected_large) < tolerance);

    printf("meanSquaredError test passed\n");
}

int main()
{
    printf("Testing meanSquaredError\n");
    test_meanSquaredError();
    return 0;
}
