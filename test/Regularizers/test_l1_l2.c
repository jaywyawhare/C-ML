#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../include/Regularizers/l1_l2.h"
#include "../../include/Core/error_codes.h"

void run_all_tests_l1_l2()
{
    int n = 3;
    float w[3] = {1.0, -1.0, 2.0};
    float dw[3] = {0, 0, 0};
    float l1_val = 0.1;
    float l2_val = 0.01;
    float threshold = 1e-6;
    float loss = l1_l2(w, dw, l1_val, l2_val, n);

    assert(fabs(loss - 0.46) < threshold);
    assert(fabs(dw[0] - (0.1 * 1.0 + 2 * 0.01 * 1.0)) < threshold);
    assert(fabs(dw[1] - (0.1 * -1.0 + 2 * 0.01 * -1.0)) < threshold);
    assert(fabs(dw[2] - (0.1 * 1.0 + 2 * 0.01 * 2.0)) < threshold);

    float w_invalid[1] = {1.0};
    float dw_invalid[1] = {0};
    loss = l1_l2(w_invalid, dw_invalid, 0.1, 0.01, 0);
    assert(loss == CM_INVALID_PARAMETER_ERROR);

    float dw_null[3] = {0, 0, 0};
    loss = l1_l2(NULL, dw_null, 0.1, 0.01, n);
    assert(loss == CM_NULL_POINTER_ERROR);

    float w_null[3] = {1.0, -1.0, 2.0};
    loss = l1_l2(w_null, NULL, 0.1, 0.01, n);
    assert(loss == CM_NULL_POINTER_ERROR);

    printf("l1_l2 test passed\n");
}

int main()
{
    printf("Testing l1_l2\n");
    run_all_tests_l1_l2();
    return CM_SUCCESS;
}
