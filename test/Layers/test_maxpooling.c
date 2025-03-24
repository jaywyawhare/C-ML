#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../include/Layers/maxpooling.h"
#include "../../include/Core/error_codes.h"

void test_maxpooling()
{
    {
        MaxPoolingLayer layer;
        int ret = initialize_maxpooling(&layer, 2, 2);
        assert(ret == CM_SUCCESS);

        ret = compute_maxpooling_output_size(8, 2, 2);
        assert(ret == 4);

        float input[8] = {1, 3, 2, 5, 0, 1, 4, 2};
        float output[4];
        ret = forward_maxpooling(&layer, input, output, 8);
        assert(ret > 0);
        assert(fabs(output[0] - 3) < 1e-6);
        assert(fabs(output[1] - 5) < 1e-6);
        assert(fabs(output[2] - 1) < 1e-6);
        assert(fabs(output[3] - 4) < 1e-6);
        printf("MaxPooling layer normal test passed\n");
    }

    {
        int ret;
        MaxPoolingLayer layer;
        ret = initialize_maxpooling(&layer, 0, 2);
        assert(ret == CM_INVALID_KERNEL_SIZE_ERROR);

        ret = initialize_maxpooling(&layer, 2, 0);
        assert(ret == CM_INVALID_STRIDE_ERROR);

        ret = initialize_maxpooling(&layer, 2, 2);
        assert(ret == CM_SUCCESS);

        ret = compute_maxpooling_output_size(-1, 2, 2);
        assert(ret == CM_INVALID_INPUT_ERROR);

        ret = forward_maxpooling(&layer, NULL, (float[4]){0}, 8);
        assert(ret == CM_NULL_POINTER_ERROR);
        ret = forward_maxpooling(&layer, (float[8]){1, 2, 3, 4, 5, 6, 7, 8}, NULL, 8);
        assert(ret == CM_NULL_POINTER_ERROR);

        ret = free_maxpooling(NULL);
        assert(ret == CM_NULL_LAYER_ERROR);
        printf("MaxPooling layer edge case tests passed\n");
    }
}

int main()
{
    printf("Testing MaxPoolingLayer\n");
    test_maxpooling();
    return CM_SUCCESS;
}
