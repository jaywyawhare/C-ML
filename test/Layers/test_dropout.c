#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../include/Layers/dropout.h"
#include "../../include/Core/error_codes.h"

void test_dropout()
{
    {
        DropoutLayer layer;
        int ret = initialize_dropout(&layer, 0.5);
        assert(ret == CM_SUCCESS);
        assert(layer.dropout_rate == 0.5);

        float input[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        float output[5];
        ret = forward_dropout(&layer, input, output, 5);
        assert(ret == CM_SUCCESS);
        for (int i = 0; i < 5; i++)
        {
            assert(output[i] == 0.0f || output[i] == input[i] / (1 - layer.dropout_rate));
        }

        float d_output[] = {0.1, 0.2, 0.3, 0.4, 0.5};
        float d_input[5] = {0};
        ret = backward_dropout(&layer, input, output, d_output, d_input, 5);
        assert(ret == CM_SUCCESS);
        for (int i = 0; i < 5; i++)
        {
            if (output[i] == 0.0f)
                assert(d_input[i] == 0.0f);
            else
                assert(fabs(d_input[i] - d_output[i] / (1 - layer.dropout_rate)) < 1e-6);
        }
        printf("Dropout layer normal test passed\n");
    }

    {
        int ret;
        ret = initialize_dropout(NULL, 0.5);
        assert(ret == CM_NULL_POINTER_ERROR);

        DropoutLayer layer;
        ret = initialize_dropout(&layer, -0.1);
        assert(ret == CM_INVALID_PARAMETER_ERROR);
        ret = initialize_dropout(&layer, 1.1);
        assert(ret == CM_INVALID_PARAMETER_ERROR);

        ret = initialize_dropout(&layer, 0.5);
        assert(ret == CM_SUCCESS);

        ret = forward_dropout(&layer, NULL, (float[5]){0}, 5);
        assert(ret == CM_NULL_POINTER_ERROR);
        ret = forward_dropout(&layer, (float[5]){1, 2, 3, 4, 5}, NULL, 5);
        assert(ret == CM_NULL_POINTER_ERROR);

        ret = backward_dropout(&layer, NULL, (float[5]){0}, (float[5]){0}, (float[5]){0}, 5);
        assert(ret == CM_NULL_POINTER_ERROR);

        printf("Dropout layer edge case tests passed\n");
    }
}

int main()
{
    printf("Testing DropoutLayer\n");
    test_dropout();
    return CM_SUCCESS;
}
