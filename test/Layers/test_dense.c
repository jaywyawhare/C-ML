#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../include/Layers/dense.h"
#include "../../include/Core/error_codes.h"

void test_dense()
{
    {
        DenseLayer layer = {NULL, NULL, 0, 0};
        int ret = initialize_dense(&layer, 3, 2);
        assert(ret == CM_SUCCESS);

        assert(layer.weights != NULL);
        assert(layer.biases != NULL);

        float input[] = {1.0, 2.0, 3.0};
        float output[2];
        ret = forward_dense(&layer, input, output);
        assert(ret == CM_SUCCESS);
        assert(output[0] != 0.0f);
        assert(output[1] != 0.0f);

        float d_output[] = {0.1, 0.2};
        float d_input[3] = {0};
        float d_weights[6] = {0};
        float d_biases[2] = {0};
        ret = backward_dense(&layer, input, output, d_output, d_input, d_weights, d_biases);
        assert(ret == CM_SUCCESS);
        for (int i = 0; i < 6; i++)
            assert(d_weights[i] != 0.0f);
        for (int i = 0; i < 2; i++)
            assert(d_biases[i] != 0.0f);
        ret = update_dense(&layer, d_weights, d_biases, 0.01);
        assert(ret == CM_SUCCESS);

        free_dense(&layer);
        printf("Dense layer normal test passed\n");
    }

    {
        DenseLayer layer;
        int ret;
        ret = initialize_dense(NULL, 3, 2);
        assert(ret == CM_NULL_POINTER_ERROR);

        layer.weights = NULL;
        layer.biases = NULL;
        layer.input_size = 3;
        layer.output_size = 2;
        ret = forward_dense(&layer, NULL, (float[2]){0});
        assert(ret == CM_NULL_POINTER_ERROR);
        ret = forward_dense(&layer, (float[3]){1, 2, 3}, NULL);
        assert(ret == CM_NULL_POINTER_ERROR);

        ret = backward_dense(&layer, NULL, (float[2]){0}, (float[2]){0},
                             (float[3]){0}, (float[6]){0}, (float[2]){0});
        assert(ret == CM_NULL_POINTER_ERROR);

        ret = update_dense(&layer, NULL, (float[2]){0}, 0.01);
        assert(ret == CM_NULL_POINTER_ERROR);

        ret = free_dense(NULL);
        assert(ret == CM_NULL_POINTER_ERROR);

        printf("Dense layer edge case tests passed\n");
    }
}

int main()
{
    printf("Testing DenseLayer\n");
    test_dense();
    return CM_SUCCESS;
}
