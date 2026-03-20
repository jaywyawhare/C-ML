#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("RetinaNet Example\n\n");

    RetinaNetConfig rcfg = cml_zoo_retinanet_default_config();
    printf("RetinaNet config:\n");
    printf("  num_classes=%d, num_anchors=%d, fpn_channels=%d\n",
           rcfg.num_classes, rcfg.num_anchors, rcfg.fpn_channels);

    Module* model = cml_zoo_retinanet_create(&rcfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!model) { printf("Failed to create RetinaNet\n"); return 1; }

    printf("\nRetinaNet created successfully.\n");

    module_free(model);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
