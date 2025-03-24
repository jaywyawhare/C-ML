#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../include/Preprocessing/standard_scaler.h"
#include "../../include/Core/error_codes.h"

void test_standard_scaler()
{
    float data[] = {1, 2, 3, 4, 5};
    int size = sizeof(data) / sizeof(data[0]);
    float *scaled = standard_scaler(data, size);
    float mean = 0, std = 0;
    for (int i = 0; i < size; i++)
    {
        mean += scaled[i];
    }
    mean /= size;
    for (int i = 0; i < size; i++)
    {
        std += pow(scaled[i] - mean, 2);
    }
    std = sqrt(std / size);
    assert(fabs(mean) < 1e-6);
    assert(fabs(std - 1) < 1e-6);
    free(scaled);

    scaled = standard_scaler(NULL, 5);
    assert(scaled == NULL);

    float data2[] = {1, 2, 3, 4, 5};
    scaled = standard_scaler(data2, 0);
    assert(scaled == NULL);

    printf("standard_scaler test passed\n");
}

int main()
{
    printf("Testing standard_scaler\n");
    test_standard_scaler();
    return CM_SUCCESS;
}
