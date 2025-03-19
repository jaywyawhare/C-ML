#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../src/my_functions.h"

void test_l1()
{
    float x = 1.0, y = 2.0;
    float lr = 0.01;
    float w = 0.5, b = 0.1;
    float v_w = 0, v_b = 0;
    float s_w = 0, s_b = 0;
    float beta1 = 0.9, beta2 = 0.999;
    float epsilon = 1e-8;
    float reg_l1 = 0.01;
    float threshold = 1e-6;

    float loss = l1(x, y, lr, w, b, v_w, v_b, s_w, s_b, beta1, beta2, epsilon);
    assert(fabs(loss - 1.96) < threshold);
    printf("l1 test passed");
}

int main()
{
    printf("Testing l1\n");
    test_l1();
    return 0;
}
