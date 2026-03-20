#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("Mask R-CNN Example\n\n");

    MaskRCNNConfig mcfg = cml_zoo_mask_rcnn_default_config();
    printf("Mask R-CNN config:\n");
    printf("  num_classes=%d, num_anchors=%d, fpn_channels=%d\n",
           mcfg.num_classes, mcfg.num_anchors, mcfg.fpn_channels);
    printf("  roi_output_size=%d, mask_output_size=%d\n",
           mcfg.roi_output_size, mcfg.mask_output_size);

    Module* model = cml_zoo_mask_rcnn_create(&mcfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!model) { printf("Failed to create Mask R-CNN\n"); return 1; }

    printf("\nMask R-CNN created successfully.\n");

    module_free(model);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
