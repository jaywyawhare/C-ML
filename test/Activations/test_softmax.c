#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
#include"../../src/Activations/softmax.h"

void test_softmax(){
    float input[4][3] = {
        {0.0, 0.0, 0.0}, 
        {1.0, 2.0, 3.0}, 
        {-1.0, -2.0, -3.0},
        {1000.0, 1001.0, 1002.0}
    };
    float expected_output[4][3] = {
        {0.33333333f, 0.33333333f, 0.33333333f},
        {0.09003057f, 0.24472847f, 0.66524096f},
        {0.66524096f, 0.24472847f, 0.09003057f},
        {0.09003057f, 0.24472847f, 0.66524096f} 
    };
    float tolerance = 1e-6;

    for(int i = 0; i < 4; i++){
        float *output = softmax(input[i], 3);
        for(int j = 0; j < 3; j++){
            assert(fabs(output[j] - expected_output[i][j]) < tolerance);
        }
        freeSoftmax(output);
    }
    printf("softmax activation function test passed\n");
}

int main(){
    printf("Testing softmax activation function\n");
    test_softmax();
    return 0;
}
