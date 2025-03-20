# Usage

The `main.c` file demonstrates how to use the library to create a simple neural network. Below is an example:

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/Layers/dense.h"
#include "../src/Activations/relu.h"
#include "../src/Loss_Functions/meanSquaredError.h"

int main()
{
    float input[] = {1.0, 2.0, 3.0};
    int input_size = 3;

    float target[] = {0.0, 1.0};
    int output_size = 2;

    DenseLayer dense_layer = {NULL, NULL, 0, 0};
    initializeDense(&dense_layer, input_size, output_size);

    float dense_output[2];
    forwardDense(&dense_layer, input, dense_output);

    for (int i = 0; i < output_size; i++)
    {
        dense_output[i] = relu(dense_output[i]);
    }

    float loss = meanSquaredError(target, dense_output, output_size);
    printf("Loss: %f\n", loss);

    freeDense(&dense_layer);
    return 0;
}
```
