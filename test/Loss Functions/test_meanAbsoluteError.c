#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../src/my_functions.h"

void test_meanAbsoluteError()
{
    float y[3] = {1.0, 2.0, 3.0};
    float yHat[3] = {1.1, 1.9, 3.2};
    float expected = 0.13333333f;
    float tolerance = 1e-6;
    assert(fabs(meanAbsoluteError(y, yHat, 3) - expected) < tolerance);

    float y_empty[0] = {};
    float yHat_empty[0] = {};
    assert(meanAbsoluteError(y_empty, yHat_empty, 0) == 0);

    printf("meanAbsoluteError test passed\n");
}

int main()
{
    printf("Testing meanAbsoluteError\n");
    test_meanAbsoluteError();
    return 0;
}
