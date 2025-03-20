#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../src/Regularizers/l1_l2.h"

void test_l1_l2()
{
    int n = 3;
    float w[3] = {1.0, -1.0, 2.0};
    float dw[3] = {0, 0, 0};
    float l1 = 0.1;
    float l2 = 0.01;
    float threshold = 1e-6;

    float loss = l1_l2(w, dw, l1, l2, n);
    assert(fabs(loss - 0.46) < threshold);
    assert(fabs(dw[0] - 0.12) < threshold);
    assert(fabs(dw[1] + 0.12) < threshold);
    assert(fabs(dw[2] - 0.14) < threshold);

    printf("l1_l2 test passed\n");
}

int main()
{
    printf("Testing l1_l2\n");
    test_l1_l2();
    return 0;
}
