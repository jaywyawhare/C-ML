#ifndef CML_ZOO_YOLOV8_H
#define CML_ZOO_YOLOV8_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_classes;
    int input_size;
    float conf_threshold;
    float nms_threshold;
    DType dtype;
    DeviceType device;
} YOLOv8Config;

YOLOv8Config yolov8n_config(int num_classes);

Module* cml_zoo_yolov8n(const YOLOv8Config* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_YOLOV8_H */
