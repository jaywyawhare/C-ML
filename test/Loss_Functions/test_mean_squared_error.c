#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../include/Loss_Functions/mean_squared_error.h"
#include "../../include/Core/error_codes.h"

void test_mean_squared_error()
{
    float y[3] = {1.0, 2.0, 3.0};
    float yHat[3] = {1.1, 1.9, 3.2};
    float expected = 0.02f;
    float tolerance = 1e-6;
    assert(fabs(mean_squared_error(y, yHat, 3) - expected) < tolerance);

    float y_empty[0] = {};
    float yHat_empty[0] = {};
    assert(mean_squared_error(y_empty, yHat_empty, 0) == CM_INVALID_INPUT_ERROR);

    float y_identical[3] = {1.0, 2.0, 3.0};
    float yHat_identical[3] = {1.0, 2.0, 3.0};
    assert(mean_squared_error(y_identical, yHat_identical, 3) == 0);

    float y_large[3] = {1e6, 2e6, 3e6};
    float yHat_large[3] = {1e6 + 1, 2e6 - 1, 3e6 + 2};
    float expected_large = (1 + 1 + 4) / 3.0f;
    assert(fabs(mean_squared_error(y_large, yHat_large, 3) - expected_large) < tolerance);

    assert(mean_squared_error(NULL, yHat, 3) == CM_INVALID_INPUT_ERROR);
    assert(mean_squared_error(y, NULL, 3) == CM_INVALID_INPUT_ERROR);

    assert(mean_squared_error(y, yHat, 0) == CM_INVALID_INPUT_ERROR);

    printf("mean_squared_error test passed\n");
}

int main()
{
    printf("Testing mean_squared_error\n");
    test_mean_squared_error();
    return CM_SUCCESS;
}
