#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("EfficientNet Example\n\n");

    EfficientNetConfig ecfg = efficientnet_b0_config(10);
    printf("EfficientNet-B0 config:\n");
    printf("  num_classes=%d, dropout=%.2f\n", ecfg.num_classes, ecfg.dropout_rate);
    printf("  width_mult=%.1f, depth_mult=%.1f\n", ecfg.width_mult, ecfg.depth_mult);

    Module* model = cml_zoo_efficientnet_b0(&ecfg);
    if (!model) { printf("Failed to create EfficientNet-B0\n"); return 1; }

    printf("\nEfficientNet-B0 created successfully.\n");
    printf("(Forward pass requires 224x224 input — skipped for speed.)\n");

    module_free(model);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
