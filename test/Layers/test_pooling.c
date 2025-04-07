#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../include/Layers/pooling.h"
#include "../../include/Core/error_codes.h"

void test_pooling()
{
    {
        PoolingLayer layer;
        int ret = initialize_pooling(&layer, 2, 2);
        assert(ret == CM_SUCCESS);

        ret = compute_pooling_output_size(8, 2, 2);
        assert(ret == 4);

        float input[8] = {1, 3, 2, 5, 0, 1, 4, 2};
        float output[4];
        ret = forward_pooling(&layer, input, output, 8);
        assert(ret == 4);
        assert(fabs(output[0] - 2) < 1e-6);
        assert(fabs(output[1] - 3.5) < 1e-6);
        assert(fabs(output[2] - 0.5) < 1e-6);
        assert(fabs(output[3] - 3) < 1e-6);

        ret = free_pooling(&layer);
        assert(ret == CM_SUCCESS);
        printf("Pooling layer normal test passed\n");
    }

    {
        int ret;
        PoolingLayer layer;
        ret = initialize_pooling(&layer, 0, 2);
        assert(ret == CM_INVALID_KERNEL_SIZE_ERROR);

        ret = initialize_pooling(&layer, 2, 0);
        assert(ret == CM_INVALID_STRIDE_ERROR);

        ret = initialize_pooling(&layer, 2, 2);
        assert(ret == CM_SUCCESS);

        ret = compute_pooling_output_size(-1, 2, 2);
        assert(ret == CM_INVALID_INPUT_ERROR);

        ret = forward_pooling(&layer, NULL, (float[4]){0}, 8);
        assert(ret == CM_NULL_POINTER_ERROR);
        ret = forward_pooling(&layer, (float[8]){1, 2, 3, 4, 5, 6, 7, 8}, NULL, 8);
        assert(ret == CM_NULL_POINTER_ERROR);

        printf("Pooling layer edge case tests passed\n");
    }
}

int main()
{
    printf("Testing PoolingLayer\n");
    test_pooling();
    return CM_SUCCESS;
}
