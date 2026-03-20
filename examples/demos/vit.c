#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("Vision Transformer (ViT) Example\n\n");

    ViTConfig vcfg = cml_zoo_vit_config_tiny();
    vcfg.num_classes = 10;
    printf("ViT-Tiny config:\n");
    printf("  image=%d, patch=%d, layers=%d, heads=%d, hidden=%d, classes=%d\n",
           vcfg.image_size, vcfg.patch_size, vcfg.n_layer, vcfg.n_head,
           vcfg.hidden_size, vcfg.num_classes);

    Module* vit = cml_zoo_vit_create(&vcfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!vit) { printf("Failed to create ViT\n"); return 1; }

    int num_patches = (vcfg.image_size / vcfg.patch_size) *
                      (vcfg.image_size / vcfg.patch_size);
    printf("  num_patches=%d\n", num_patches);

    printf("\nViT-Tiny created successfully.\n");
    printf("(Forward pass requires %dx%d input — skipped for speed.)\n",
           vcfg.image_size, vcfg.image_size);

    module_free(vit);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
