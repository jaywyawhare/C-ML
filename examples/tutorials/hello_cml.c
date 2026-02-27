/**
 * Hello World Example for C-ML
 *
 * This is the simplest possible C-ML program to verify installation.
 *
 * Compile:
 *   gcc hello_cml.c -I/usr/local/include/cml -lcml -lm -o hello_cml
 *
 * Run:
 *   ./hello_cml
 */

#include <cml/cml.h>
#include <stdio.h>

int main() {
    printf("=== C-ML Hello World ===\n\n");

    // Initialize C-ML
    cml_init();
    printf("✓ C-ML initialized successfully!\n\n");

    // Create a simple tensor
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[]  = {2, 2};
    Tensor* t    = cml_tensor(data, shape, 2, NULL);
    printf("✓ Created a 2x2 tensor\n\n");

    // Create a simple neural network
    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(10, 5, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(5, 2, DTYPE_FLOAT32, DEVICE_CPU, true));
    printf("✓ Created a neural network (10 -> 5 -> 2)\n\n");

    // Print model summary
    printf("Model Summary:\n");
    cml_summary((Module*)model);

    printf("\n=== All tests passed! ===\n");
    printf("C-ML is working correctly on your system.\n\n");

    // Cleanup
    cml_cleanup();

    return 0;
}
