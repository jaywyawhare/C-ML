#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../include/Metrics/mean_absolute_error.h"
#include "../../include/Core/error_codes.h"

void test_mean_absolute_error()
{
    float y[3] = {1.0, 2.0, 3.0};
    float yHat[3] = {1.1, 1.9, 3.2};
    float expected = 0.13333333f;
    float tolerance = 1e-6;
    assert(fabs(mean_absolute_error(y, yHat, 3) - expected) < tolerance);

    float y_empty[0] = {};
    float yHat_empty[0] = {};
    assert(mean_absolute_error(y_empty, yHat_empty, 0) == CM_INVALID_INPUT_ERROR);

    assert(mean_absolute_error(NULL, yHat, 3) == CM_INVALID_INPUT_ERROR);
    assert(mean_absolute_error(y, NULL, 3) == CM_INVALID_INPUT_ERROR);

    assert(mean_absolute_error(y, yHat, 0) == CM_INVALID_INPUT_ERROR);

    printf("mean_absolute_error test passed\n");
}

int main()
{
    printf("Testing mean_absolute_error\n");
    test_mean_absolute_error();
    return CM_SUCCESS;
}
