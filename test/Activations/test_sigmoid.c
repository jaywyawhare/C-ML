#include<stdio.h>
#include<assert.h>
#include<math.h>
#include"../../src/my_functions.h"

void test_sigmoid(){
    float input[7] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e6, -1e6};
    float expected_output[7] = {0.5f, 0.73105858f, 0.26894142f, 0.88079708f, 0.11920292f, 1.0f, 0.0f};
    float tolerance = 1e-6;
    for(int i = 0; i < 7; i++){
        assert(fabs(sigmoid(input[i]) - expected_output[i]) < tolerance);
    }
    printf("sigmoid activation function test passed\n");
}

int main(){
    printf("Testing sigmoid activation function\n");
    test_sigmoid();
    return 0;
}
