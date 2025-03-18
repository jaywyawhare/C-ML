#include<stdio.h>
#include<assert.h>
#include<math.h>
#include"../../src/my_functions.h"


void test_elu(){
    float input[7] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e6, -1e6};
    float alpha = 1.0;
    float expected_output[7] = {0.0f, 1.0f, -0.63212055f, 2.0f, -0.86466472f, 1e6f, -1.0f}; // Adjusted for large inputs
    float tolerance = 1e-6;
    for(int i = 0; i < 7; i++){
        assert(fabs(elu(input[i], alpha) - expected_output[i]) < tolerance);
    }
    printf("elu activation function test passed\n");
}

int main(){
    printf("Testing elu activation function\n");
    test_elu();
    return 0;
}