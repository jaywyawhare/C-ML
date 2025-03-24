#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../include/Regularizers/l1.h"
#include "../../include/Core/error_codes.h"

void run_all_tests_l1()
{
    float x = 1.0, y = 2.0;
    float lr = 0.01;
    float w = 0.5, b = 0.1;
    float v_w = 0, v_b = 0;
    float s_w = 0, s_b = 0;
    float beta1 = 0.9, beta2 = 0.999;
    float epsilon = 1e-8;
    float threshold = 1e-6;

    float loss = l1(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon);
    assert(fabs(loss - (pow((0.5 * 1.0 + 0.1 - 2.0), 2) + fabs(0.5))) < threshold);

    loss = l1(x, y, lr, NULL, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon);
    assert(loss == CM_NULL_POINTER_ERROR);

    loss = l1(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, 0);
    assert(loss == CM_INVALID_PARAMETER_ERROR);
    printf("l1 test passed\n");
}

int main()
{
    printf("Testing l1\n");
    run_all_tests_l1();
    return CM_SUCCESS;
}
