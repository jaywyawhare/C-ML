/**
 * @file augmentation.h
 * @brief Data augmentation functions for image and tensor data
 */

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

/**
 * @brief Create augmentation configuration
 *
 * @return New augmentation config with defaults
 */
AugmentationConfig* augmentation_config_create(void);

/**
 * @brief Free augmentation configuration
 *
 * @param config Config to free
 */
void augmentation_config_free(AugmentationConfig* config);

// Image Augmentation Functions

/**
 * @brief Random crop tensor
 *
 * @param input Input tensor [batch, channels, height, width]
 * @param crop_height Crop height
 * @param crop_width Crop width
 * @return Cropped tensor, or NULL on failure
 */
Tensor* augment_random_crop(Tensor* input, int crop_height, int crop_width);

/**
 * @brief Random horizontal flip
 *
 * @param input Input tensor [batch, channels, height, width]
 * @param prob Probability of flipping (0.0 to 1.0)
 * @return Flipped tensor, or NULL on failure
 */
Tensor* augment_random_horizontal_flip(Tensor* input, float prob);

/**
 * @brief Random vertical flip
 *
 * @param input Input tensor [batch, channels, height, width]
 * @param prob Probability of flipping (0.0 to 1.0)
 * @return Flipped tensor, or NULL on failure
 */
Tensor* augment_random_vertical_flip(Tensor* input, float prob);

/**
 * @brief Random rotation
 *
 * @param input Input tensor [batch, channels, height, width]
 * @param angle_min Minimum rotation angle (degrees)
 * @param angle_max Maximum rotation angle (degrees)
 * @return Rotated tensor, or NULL on failure
 */
Tensor* augment_random_rotation(Tensor* input, float angle_min, float angle_max);

/**
 * @brief Color jitter (brightness, contrast, saturation, hue)
 *
 * @param input Input tensor [batch, channels, height, width]
 * @param brightness Brightness factor
 * @param contrast Contrast factor
 * @param saturation Saturation factor
 * @param hue Hue factor
 * @return Jittered tensor, or NULL on failure
 */
Tensor* augment_color_jitter(Tensor* input, float brightness, float contrast, float saturation,
                             float hue);

/**
 * @brief Normalize tensor
 *
 * @param input Input tensor [batch, channels, height, width]
 * @param mean Mean values per channel
 * @param std Standard deviation values per channel
 * @param num_channels Number of channels
 * @return Normalized tensor, or NULL on failure
 */
Tensor* augment_normalize(Tensor* input, float* mean, float* std, int num_channels);

/**
 * @brief Apply augmentation pipeline
 *
 * @param input Input tensor
 * @param config Augmentation configuration
 * @return Augmented tensor, or NULL on failure
 */
Tensor* augment_apply(Tensor* input, AugmentationConfig* config);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_AUGMENTATION_H
