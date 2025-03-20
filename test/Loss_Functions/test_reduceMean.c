#include <stdio.h>
#include <assert.h>
#include "../../src/Loss_Functions/reduceMean.h"

void test_reduceMean()
{
    float loss[3] = {1.0, 2.0, 3.0};
    float expected = 2.0f;
    assert(reduceMean(loss, 3) == expected);

    float loss_empty[0] = {};
    assert(reduceMean(loss_empty, 0) == 0);

    float loss_identical[3] = {5.0, 5.0, 5.0};
    assert(reduceMean(loss_identical, 3) == 5.0f);

    printf("reduceMean test passed\n");
}

int main()
{
    printf("Testing reduceMean\n");
    test_reduceMean();
    return 0;
}
