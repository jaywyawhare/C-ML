#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../include/Preprocessing/min_max_scaler.h"
#include "../../include/Core/error_codes.h"

void test_min_max_scaler()
{
    float data[] = {10, 20, 30, 40, 50};
    int size = sizeof(data) / sizeof(data[0]);
    float *scaled = min_max_scaler(data, size);
    for (int i = 0; i < size; i++)
    {
        float expected = (data[i] - 10) / 40.0f;
        assert(fabs(scaled[i] - expected) < 1e-6);
    }
    free(scaled);

    scaled = min_max_scaler(NULL, 5);
    assert(scaled == NULL);

    float data2[] = {10, 20, 30, 40, 50};
    scaled = min_max_scaler(data2, 0);
    assert(scaled == NULL);

    printf("min_max_scaler test passed\n");
}

int main()
{
    printf("Testing min_max_scaler\n");
    test_min_max_scaler();
    return CM_SUCCESS;
}
