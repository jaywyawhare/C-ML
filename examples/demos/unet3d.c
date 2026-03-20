#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("U-Net 3D Example\n\n");

    CMLUNet3DConfig ucfg = cml_zoo_unet3d_config_default();
    printf("U-Net 3D config:\n");
    printf("  in_channels=%d, num_classes=%d, depth=%d, base_filters=%d\n",
           ucfg.in_channels, ucfg.num_classes, ucfg.depth, ucfg.base_filters);

    Module* model = cml_zoo_unet3d_create(&ucfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!model) { printf("Failed to create U-Net 3D\n"); return 1; }

    printf("\nU-Net 3D created successfully.\n");
    printf("(Suitable for 3D medical image segmentation.)\n");

    module_free(model);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
