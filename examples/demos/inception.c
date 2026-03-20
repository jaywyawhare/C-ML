#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("Inception V3 Example\n\n");

    int num_classes = 10;
    Module* model = cml_zoo_inception_v3_create(num_classes, DTYPE_FLOAT32, DEVICE_CPU);
    if (!model) { printf("Failed to create Inception V3\n"); return 1; }

    printf("Inception V3 (%d classes) created successfully.\n", num_classes);
    printf("(Forward pass requires 299x299 input — skipped for speed.)\n");

    module_free(model);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
