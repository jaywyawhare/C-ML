#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("YOLOv8 Example\n\n");

    YOLOv8Config ycfg = yolov8n_config(80);
    printf("YOLOv8n config:\n");
    printf("  num_classes=%d, input_size=%d\n", ycfg.num_classes, ycfg.input_size);
    printf("  conf_threshold=%.2f, nms_threshold=%.2f\n",
           ycfg.conf_threshold, ycfg.nms_threshold);

    Module* model = cml_zoo_yolov8n(&ycfg);
    if (!model) { printf("Failed to create YOLOv8n\n"); return 1; }

    printf("\nYOLOv8n (80 classes, COCO) created successfully.\n");

    module_free(model);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
