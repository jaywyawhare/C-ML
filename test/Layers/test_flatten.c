#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../include/Layers/flatten.h"
#include "../../include/Core/error_codes.h"

void test_flatten()
{
    {
        FlattenLayer layer = {0, 0};
        int ret = initialize_flatten(&layer, 3);
        assert(ret == CM_SUCCESS);

        float input[] = {1.0, 2.0, 3.0};
        float output[3];
        ret = forward_flatten(&layer, input, output);
        assert(ret == CM_SUCCESS);
        for (int i = 0; i < 3; i++)
            assert(output[i] == input[i]);

        float d_output[] = {0.5, 0.5, 0.5};
        float d_input[3] = {0};
        ret = backward_flatten(&layer, input, output, d_output, d_input);
        assert(ret == CM_SUCCESS);
        for (int i = 0; i < 3; i++)
            assert(d_input[i] == d_output[i]);

        ret = free_flatten(&layer);
        assert(ret == CM_SUCCESS);
        printf("Flatten layer normal test passed\n");
    }

    {
        int ret;
        ret = initialize_flatten(NULL, 3);
        assert(ret == CM_NULL_LAYER_ERROR);

        FlattenLayer layer = {0, 0};
        ret = initialize_flatten(&layer, 0);
        assert(ret == CM_INVALID_LAYER_DIMENSIONS_ERROR);

        ret = initialize_flatten(&layer, 3);
        assert(ret == CM_SUCCESS);

        ret = forward_flatten(&layer, NULL, (float[3]){0});
        assert(ret == CM_NULL_POINTER_ERROR);
        ret = forward_flatten(&layer, (float[3]){1, 2, 3}, NULL);
        assert(ret == CM_NULL_POINTER_ERROR);

        ret = backward_flatten(&layer, NULL, (float[3]){0}, (float[3]){0}, (float[3]){0});
        assert(ret == CM_NULL_POINTER_ERROR);

        printf("Flatten layer edge case tests passed\n");
    }
}

int main()
{
    printf("Testing FlattenLayer\n");
    test_flatten();
    return CM_SUCCESS;
}
