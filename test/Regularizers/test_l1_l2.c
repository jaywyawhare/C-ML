#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../src/my_functions.h"

void test_l1_l2()
{
    int n = 3;
    float w_arr[3] = {1.0, -1.0, 2.0};
    float dw[3] = {0, 0, 0};
    float dw_l1[3] = {0, 0, 0};
    float dw_l2[3] = {0, 0, 0};
    float reg_l1 = 0.1;
    float reg_l2 = 0.01;
    float threshold = 1e-6;

    float loss = l1_l2(w_arr, dw, dw_l1, dw_l1, dw_l2, dw_l2, reg_l1, reg_l2, n);
    assert(fabs(loss - 0.46) < threshold);
    assert(fabs(dw[0] - 0.12) < threshold);
    assert(fabs(dw[1] + 0.12) < threshold);
    assert(fabs(dw[2] - 0.14) < threshold);
    assert(dw_l1[0] == 1);
    assert(dw_l1[1] == -1);
    assert(dw_l1[2] == 1);
    assert(fabs(dw_l2[0] - 2.0) < threshold);
    assert(fabs(dw_l2[1] + 2.0) < threshold);
    assert(fabs(dw_l2[2] - 4.0) < threshold);

    printf("l1_l2 test passed");
}

int main()
{
    printf("Testing l1_l2\n");
    test_l1_l2();
    return 0;
}
