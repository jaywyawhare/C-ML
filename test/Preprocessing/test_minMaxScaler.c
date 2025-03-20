#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../src/Preprocessing/minMaxScaler.h"

void test_minMaxScaler_normal_case()
{
    float data[] = {10, 20, 30, 40, 50};
    int size = sizeof(data) / sizeof(data[0]);
    float *scaled = minMaxScaler(data, size);
    for (int i = 0; i < size; i++)
    {
        float expected = (data[i] - 10) / 40.0f;
        assert(fabs(scaled[i] - expected) < 1e-6);
    }
    free(scaled);
    printf("minMaxScaler test passed\n");
}

int main()
{
    printf("Testing minMaxScaler\n");
    test_minMaxScaler_normal_case();
    return 0;
}
