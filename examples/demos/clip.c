#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("CLIP Example\n\n");

    CMLCLIPConfig ccfg = cml_zoo_clip_config_vit_b32();
    printf("CLIP ViT-B/32 config:\n");
    printf("  image=%d, patch=%d\n", ccfg.image_size, ccfg.patch_size);
    printf("  vision: layers=%d, heads=%d, dim=%d\n",
           ccfg.vision_layers, ccfg.vision_heads, ccfg.vision_dim);
    printf("  text:   layers=%d, heads=%d, dim=%d\n",
           ccfg.text_layers, ccfg.text_heads, ccfg.text_dim);
    printf("  vocab=%d, embed_dim=%d\n", ccfg.vocab_size, ccfg.embed_dim);

    Module* clip = cml_zoo_clip_create(&ccfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!clip) { printf("Failed to create CLIP\n"); return 1; }

    printf("\nCLIP ViT-B/32 created successfully.\n");
    printf("(Forward pass requires 224x224 image — skipped for speed.)\n");

    module_free(clip);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
