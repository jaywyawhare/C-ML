#include<stdio.h>
#include<assert.h>
#include<math.h>
#include"../../src/Activations/leakyRelu.h"

void test_leakyRelu(){
    float input[7] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e6, -1e6};
    float expected_output[7] = {0.0f, 1.0f, -0.01f, 2.0f, -0.02f, 1e6f, -1e4f};
    float tolerance = 1e-6;
    for(int i = 0; i < 7; i++){
        assert(fabs(leakyRelu(input[i]) - expected_output[i]) < tolerance);
    }
    printf("leakyRelu activation function test passed\n");
}

int main(){
    printf("Testing leakyRelu activation function\n");
    test_leakyRelu();
    return 0;
}