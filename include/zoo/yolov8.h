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
YOLOv8Config yolov8s_config(int num_classes);
YOLOv8Config yolov8m_config(int num_classes);
YOLOv8Config yolov8l_config(int num_classes);
YOLOv8Config yolov8x_config(int num_classes);

Module* cml_zoo_yolov8n(const YOLOv8Config* config);
Module* cml_zoo_yolov8s(const YOLOv8Config* config);
Module* cml_zoo_yolov8m(const YOLOv8Config* config);
Module* cml_zoo_yolov8l(const YOLOv8Config* config);
Module* cml_zoo_yolov8x(const YOLOv8Config* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_YOLOV8_H */
