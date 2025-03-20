#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../src/Regularizers/l2.h"

void test_l2()
{
    float x = 1.0, y = 2.0;
    float lr = 0.01;
    float w = 0.5, b = 0.1;
    float v_w = 0, v_b = 0;
    float s_w = 0, s_b = 0;
    float beta1 = 0.9, beta2 = 0.999;
    float epsilon = 1e-8;
    float reg_l2 = 0.01;
    float threshold = 1e-6;

    float loss = l2(x, y, lr, &w, &b, &v_w, &v_b, &s_w, &s_b, beta1, beta2, epsilon, reg_l2);
    assert(fabs(loss - (1.96 + reg_l2 * pow(0.5, 2))) < threshold); // Include L2 regularization in the expected loss
    printf("l2 test passed\n");
}

int main()
{
    printf("Testing l2\n");
    test_l2();
    return 0;
}
