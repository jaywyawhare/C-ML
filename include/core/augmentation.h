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

/**
 * All augmentation functions expect input shape [batch, channels, height, width]
 * @param input (Tensor*)
 * @param crop_height (int)
 * @param crop_width (int)
 * @return (Tensor*)
 */
Tensor* augment_random_crop(Tensor* input, int crop_height, int crop_width);
Tensor* augment_random_horizontal_flip(Tensor* input, float prob);
Tensor* augment_random_vertical_flip(Tensor* input, float prob);
Tensor* augment_random_rotation(Tensor* input, float angle_min, float angle_max);
Tensor* augment_color_jitter(Tensor* input, float brightness, float contrast, float saturation,
                             float hue);
Tensor* augment_normalize(Tensor* input, float* mean, float* std, int num_channels);
Tensor* augment_apply(Tensor* input, AugmentationConfig* config);

/**
 * Zero a random size x size square per image (cutout / random erasing).
 * @param input (Tensor*)
 * @param size (int)
 * @return (Tensor*)
 */
Tensor* augment_cutout(Tensor* input, int size);

/**
 * Mixup convex blend out = lambda*a + (1-lambda)*b of two identically-shaped batches.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @param lambda (float)
 * @return (Tensor*)
 */
Tensor* augment_mixup(Tensor* a, Tensor* b, float lambda);

/**
 * CutMix: paste b's box [top, top+box_h) x [left, left+box_w) over a copy of a
 * (label-mix weight lambda = 1 - box_area/(H*W)).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @param top (int)
 * @param left (int)
 * @param box_h (int)
 * @param box_w (int)
 * @return (Tensor*)
 */
Tensor* augment_cutmix(Tensor* a, Tensor* b, int top, int left, int box_h, int box_w);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_AUGMENTATION_H
