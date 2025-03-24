#include <stdio.h>
#include <assert.h>
#include "../../include/Loss_Functions/reduce_mean.h"
#include "../../include/Core/error_codes.h"

void test_reduce_mean()
{
    float loss[3] = {1.0, 2.0, 3.0};
    float expected = 2.0f;
    assert(reduce_mean(loss, 3) == expected);

    float loss_empty[0] = {};
    assert(reduce_mean(loss_empty, 0) == CM_INVALID_INPUT_ERROR);

    float loss_identical[3] = {5.0, 5.0, 5.0};
    assert(reduce_mean(loss_identical, 3) == 5.0f);

    assert(reduce_mean(NULL, 3) == CM_INVALID_INPUT_ERROR);

    assert(reduce_mean(loss, 0) == CM_INVALID_INPUT_ERROR);

    printf("reduce_mean test passed\n");
}

int main()
{
    printf("Testing reduce_mean\n");
    test_reduce_mean();
    return CM_SUCCESS;
}
