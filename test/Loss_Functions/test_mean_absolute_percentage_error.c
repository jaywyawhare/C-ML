#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../include/Loss_Functions/mean_absolute_percentage_error.h"
#include "../../include/Core/error_codes.h"

void test_mean_absolute_percentage_error()
{
    float y[3] = {1.0, 2.0, 3.0};
    float yHat[3] = {1.1, 1.9, 3.2};
    float expected = 7.2222222f;
    float tolerance = 1e-5;
    assert(fabs(mean_absolute_percentage_error(y, yHat, 3) - expected) < tolerance);

    float y_zero[3] = {0.0, 2.0, 3.0};
    float yHat_zero[3] = {1.0, 2.0, 3.0};
    float expected_zero = 0.0f;
    assert(fabs(mean_absolute_percentage_error(y_zero, yHat_zero, 3) - expected_zero) < tolerance);

    float y_large[3] = {1e6, 2e6, 3e6};
    float yHat_large[3] = {1e6 + 1, 2e6 - 1, 3e6 + 2};
    float expected_large = ((1.0 / 1e6 + 1.0 / 2e6 + 2.0 / 3e6) / 3) * 100;
    assert(fabs(mean_absolute_percentage_error(y_large, yHat_large, 3) - expected_large) < tolerance);

    assert(mean_absolute_percentage_error(NULL, yHat, 3) == CM_INVALID_INPUT_ERROR);
    assert(mean_absolute_percentage_error(y, NULL, 3) == CM_INVALID_INPUT_ERROR);

    assert(mean_absolute_percentage_error(y, yHat, 0) == CM_INVALID_INPUT_ERROR);

    printf("mean_absolute_percentage_error test passed\n");
}

int main()
{
    printf("Testing mean_absolute_percentage_error\n");
    test_mean_absolute_percentage_error();
    return CM_SUCCESS;
}
