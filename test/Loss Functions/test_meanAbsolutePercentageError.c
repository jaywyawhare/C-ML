#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../src/my_functions.h"

void test_meanAbsolutePercentageError()
{
    float y[3] = {1.0, 2.0, 3.0};
    float yHat[3] = {1.1, 1.9, 3.2};
    float expected = 7.2222222f;
    float tolerance = 1e-5;
    assert(fabs(meanAbsolutePercentageError(y, yHat, 3) - expected) < tolerance);

    float y_zero[3] = {0.0, 2.0, 3.0};
    float yHat_zero[3] = {1.0, 2.0, 3.0};
    float expected_zero = 0.0f;
    assert(fabs(meanAbsolutePercentageError(y_zero, yHat_zero, 3) - expected_zero) < tolerance);

    float y_large[3] = {1e6, 2e6, 3e6};
    float yHat_large[3] = {1e6 + 1, 2e6 - 1, 3e6 + 2};
    float expected_large = ((1.0 / 1e6 + 1.0 / 2e6 + 2.0 / 3e6) / 3) * 100;
    assert(fabs(meanAbsolutePercentageError(y_large, yHat_large, 3) - expected_large) < tolerance);

    printf("meanAbsolutePercentageError test passed\n");
}

int main()
{
    printf("Testing meanAbsolutePercentageError\n");
    test_meanAbsolutePercentageError();
    return 0;
}
