#ifndef CML_CORE_AUGMENTATION_H
#define CML_CORE_AUGMENTATION_H

#include "tensor/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AugmentationConfig {
    bool random_crop;
    int crop_size[2];

    bool random_horizontal_flip;
    float horizontal_flip_prob;
    bool random_vertical_flip;
    float vertical_flip_prob;

    bool random_rotation;
    float rotation_angle_min;
    float rotation_angle_max;

    bool color_jitter;
    float brightness;
    float contrast;
    float saturation;
    float hue;

    bool normalize;
    float* mean;
    float* std;
    int num_channels;
} AugmentationConfig;

AugmentationConfig* augmentation_config_create(void);
void augmentation_config_free(AugmentationConfig* config);

/* All augmentation functions expect input shape [batch, channels, height, width] */
Tensor* augment_random_crop(Tensor* input, int crop_height, int crop_width);
Tensor* augment_random_horizontal_flip(Tensor* input, float prob);
Tensor* augment_random_vertical_flip(Tensor* input, float prob);
Tensor* augment_random_rotation(Tensor* input, float angle_min, float angle_max);
Tensor* augment_color_jitter(Tensor* input, float brightness, float contrast, float saturation,
                             float hue);
Tensor* augment_normalize(Tensor* input, float* mean, float* std, int num_channels);
Tensor* augment_apply(Tensor* input, AugmentationConfig* config);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_AUGMENTATION_H
