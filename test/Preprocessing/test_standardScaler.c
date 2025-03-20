#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../src/Preprocessing/standardScaler.h"

void test_standardScaler_normal_case()
{
    float data[] = {1, 2, 3, 4, 5};
    int size = sizeof(data) / sizeof(data[0]);
    float *scaled = standardScaler(data, size);
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
    printf("standerdScaler test passed\n");
}

int main()
{
    printf("Testing standardScaler\n");
    test_standardScaler_normal_case();
    return 0;
}
