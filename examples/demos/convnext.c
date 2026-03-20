#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("ConvNeXt Example\n\n");

    ConvNeXtConfig ccfg = cml_zoo_convnext_config_tiny();
    printf("ConvNeXt-Tiny config:\n");
    printf("  dims:   [%d, %d, %d, %d]\n",
           ccfg.dims[0], ccfg.dims[1], ccfg.dims[2], ccfg.dims[3]);
    printf("  depths: [%d, %d, %d, %d]\n",
           ccfg.depths[0], ccfg.depths[1], ccfg.depths[2], ccfg.depths[3]);

    int num_classes = 10;
    Module* model = cml_zoo_convnext_create(&ccfg, num_classes, DTYPE_FLOAT32, DEVICE_CPU);
    if (!model) { printf("Failed to create ConvNeXt\n"); return 1; }

    printf("\nConvNeXt-Tiny (%d classes) created successfully.\n", num_classes);
    printf("(Forward pass requires 224x224 input — skipped for speed.)\n");

    module_free(model);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
