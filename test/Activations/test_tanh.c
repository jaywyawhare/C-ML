#include<stdio.h>
#include<assert.h>
#include<math.h>
#include"../../src/my_functions.h"

void test_tanh(){
    float input[7] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e6, -1e6};
    float expected_output[7] = {0.0f, 0.76159416f, -0.76159416f, 0.96402758f, -0.96402758f, 1.0f, -1.0f};
    float tolerance = 1e-6;
    for(int i = 0; i < 7; i++){
        assert(fabs(tanH(input[i]) - expected_output[i]) < tolerance);
    }
    printf("tanh activation function test passed\n");
}

int main(){
    printf("Testing tanh activation function\n");
    test_tanh();
    return 0;
}
