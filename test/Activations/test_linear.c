#include<stdio.h>
#include<assert.h>
#include<math.h>
#include"../../src/my_functions.h"

void test_linear(){
    float input[7] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e6, -1e6};
    float expected_output[7] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 1e6f, -1e6f};
    float tolerance = 1e-6;
    for(int i = 0; i < 7; i++){
        assert(fabs(linear(input[i]) - expected_output[i]) < tolerance);
    }
    printf("linear activation function test passed\n");
}

int main(){
    printf("Testing linear activation function\n");
    test_linear();
    return 0;
}