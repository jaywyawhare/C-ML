/**
 * @file augmentation.c
 * @brief Data augmentation implementation
 */

#include "Core/augmentation.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Augmentation Configuration

AugmentationConfig* augmentation_config_create(void) {
    AugmentationConfig* config = CM_MALLOC(sizeof(AugmentationConfig));
    if (!config)
        return NULL;

    // Initialize defaults
    config->random_crop  = false;
    config->crop_size[0] = 0;
    config->crop_size[1] = 0;

    config->random_horizontal_flip = false;
    config->horizontal_flip_prob   = 0.5f;
    config->random_vertical_flip   = false;
    config->vertical_flip_prob     = 0.5f;

    config->random_rotation    = false;
    config->rotation_angle_min = 0.0f;
    config->rotation_angle_max = 0.0f;

    config->color_jitter = false;
    config->brightness   = 0.0f;
    config->contrast     = 0.0f;
    config->saturation   = 0.0f;
    config->hue          = 0.0f;

    config->normalize    = false;
    config->mean         = NULL;
    config->std          = NULL;
    config->num_channels = 0;

    return config;
}

void augmentation_config_free(AugmentationConfig* config) {
    if (!config)
        return;

    if (config->mean)
        CM_FREE(config->mean);
    if (config->std)
        CM_FREE(config->std);
    CM_FREE(config);
}

// Random number generator (simple linear congruential)
static unsigned int rng_seed = 1;
static void set_seed(unsigned int seed) { rng_seed = seed; }
static float rand_float(void) {
    rng_seed = (rng_seed * 1103515245 + 12345) & 0x7fffffff;
    return (float)rng_seed / 2147483648.0f;
}

// Random Crop
Tensor* augment_random_crop(Tensor* input, int crop_height, int crop_width) {
    if (!input || input->ndim != 4) {
        LOG_ERROR("Random crop requires 4D tensor [batch, channels, height, width]");
        return NULL;
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    if (crop_height > height || crop_width > width) {
        LOG_ERROR("Crop size exceeds input size");
        return NULL;
    }

    // Initialize seed if not already done
    static bool seed_set = false;
    if (!seed_set) {
        set_seed((unsigned int)time(NULL));
        seed_set = true;
    }

    // Calculate random offset
    int h_offset = (int)(rand_float() * (height - crop_height));
    int w_offset = (int)(rand_float() * (width - crop_width));

    // Create output tensor
    int output_shape[] = {batch, channels, crop_height, crop_width};
    Tensor* output     = tensor_empty(output_shape, 4, input->dtype, input->device);
    if (!output)
        return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);

    // Copy cropped region
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < crop_height; h++) {
                for (int w = 0; w < crop_width; w++) {
                    int in_idx = b * channels * height * width + c * height * width +
                                 (h_offset + h) * width + (w_offset + w);
                    int out_idx = b * channels * crop_height * crop_width +
                                  c * crop_height * crop_width + h * crop_width + w;
                    out_data[out_idx] = in_data[in_idx];
                }
            }
        }
    }

    return output;
}

// Random Horizontal Flip
Tensor* augment_random_horizontal_flip(Tensor* input, float prob) {
    if (!input || input->ndim != 4) {
        LOG_ERROR("Horizontal flip requires 4D tensor [batch, channels, height, width]");
        return NULL;
    }

    // Initialize seed if not already done
    static bool seed_set = false;
    if (!seed_set) {
        set_seed((unsigned int)time(NULL));
        seed_set = true;
    }

    // Decide whether to flip
    bool should_flip = rand_float() < prob;

    if (!should_flip) {
        return tensor_clone(input);
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    // Create output tensor
    Tensor* output = tensor_empty(input->shape, 4, input->dtype, input->device);
    if (!output)
        return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);

    // Flip horizontally
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int in_idx = b * channels * height * width + c * height * width + h * width + w;
                    int out_idx = b * channels * height * width + c * height * width + h * width +
                                  (width - 1 - w);
                    out_data[out_idx] = in_data[in_idx];
                }
            }
        }
    }

    return output;
}

// Random Vertical Flip
Tensor* augment_random_vertical_flip(Tensor* input, float prob) {
    if (!input || input->ndim != 4) {
        LOG_ERROR("Vertical flip requires 4D tensor [batch, channels, height, width]");
        return NULL;
    }

    // Initialize seed if not already done
    static bool seed_set = false;
    if (!seed_set) {
        set_seed((unsigned int)time(NULL));
        seed_set = true;
    }

    // Decide whether to flip
    bool should_flip = rand_float() < prob;

    if (!should_flip) {
        return tensor_clone(input);
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    // Create output tensor
    Tensor* output = tensor_empty(input->shape, 4, input->dtype, input->device);
    if (!output)
        return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);

    // Flip vertically
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int in_idx = b * channels * height * width + c * height * width + h * width + w;
                    int out_idx = b * channels * height * width + c * height * width +
                                  (height - 1 - h) * width + w;
                    out_data[out_idx] = in_data[in_idx];
                }
            }
        }
    }

    return output;
}

// Random Rotation (simplified - 90 degree rotations)
Tensor* augment_random_rotation(Tensor* input, float angle_min, float angle_max) {
    if (!input || input->ndim != 4) {
        LOG_ERROR("Rotation requires 4D tensor [batch, channels, height, width]");
        return NULL;
    }

    // Initialize seed if not already done
    static bool seed_set = false;
    if (!seed_set) {
        set_seed((unsigned int)time(NULL));
        seed_set = true;
    }

    // Generate random angle
    float angle = angle_min + rand_float() * (angle_max - angle_min);

    // For simplicity, support 0, 90, 180, 270 degree rotations
    int rotation = (int)(angle / 90.0f) % 4;

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    // Create output tensor
    Tensor* output = tensor_empty(input->shape, 4, input->dtype, input->device);
    if (!output)
        return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);

    // Apply rotation
    if (rotation == 0) {
        // No rotation - just copy
        memcpy(out_data, in_data, input->numel * sizeof(float));
    } else {
        // For 90/180/270 rotations, transpose and flip
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int in_idx =
                            b * channels * height * width + c * height * width + h * width + w;
                        int out_idx;

                        if (rotation == 1) { // 90 degrees
                            out_idx = b * channels * height * width + c * height * width +
                                      w * height + (height - 1 - h);
                        } else if (rotation == 2) { // 180 degrees
                            out_idx = b * channels * height * width + c * height * width +
                                      (height - 1 - h) * width + (width - 1 - w);
                        } else { // 270 degrees
                            out_idx = b * channels * height * width + c * height * width +
                                      (width - 1 - w) * height + h;
                        }

                        out_data[out_idx] = in_data[in_idx];
                    }
                }
            }
        }
    }

    return output;
}

// Color Jitter (simplified - brightness only)
Tensor* augment_color_jitter(Tensor* input, float brightness, float contrast, float saturation,
                             float hue) {
    (void)saturation; // Not implemented yet
    (void)hue;        // Not implemented yet
    if (!input || input->ndim != 4) {
        LOG_ERROR("Color jitter requires 4D tensor [batch, channels, height, width]");
        return NULL;
    }

    // Initialize seed if not already done
    static bool seed_set = false;
    if (!seed_set) {
        set_seed((unsigned int)time(NULL));
        seed_set = true;
    }

    // Generate random factors
    float bright_factor   = 1.0f + (rand_float() * 2.0f - 1.0f) * brightness;
    float contrast_factor = 1.0f + (rand_float() * 2.0f - 1.0f) * contrast;

    // Create output tensor
    Tensor* output = tensor_empty(input->shape, 4, input->dtype, input->device);
    if (!output)
        return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);

    // Apply jitter
    for (size_t i = 0; i < input->numel; i++) {
        float val = in_data[i];
        // Apply brightness
        val = val * bright_factor;
        // Apply contrast (simplified)
        val = (val - 0.5f) * contrast_factor + 0.5f;
        // Clamp to [0, 1]
        if (val < 0.0f)
            val = 0.0f;
        if (val > 1.0f)
            val = 1.0f;
        out_data[i] = val;
    }

    return output;
}

// Normalize
Tensor* augment_normalize(Tensor* input, float* mean, float* std, int num_channels) {
    if (!input || !mean || !std || input->ndim != 4) {
        LOG_ERROR("Normalize requires 4D tensor and valid mean/std arrays");
        return NULL;
    }

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    if (channels != num_channels) {
        LOG_ERROR("Number of channels doesn't match mean/std arrays");
        return NULL;
    }

    // Create output tensor
    Tensor* output = tensor_empty(input->shape, 4, input->dtype, input->device);
    if (!output)
        return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);

    // Normalize: (x - mean) / std
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            float m = mean[c];
            float s = std[c];
            if (s == 0.0f)
                s = 1.0f; // Avoid division by zero

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = b * channels * height * width + c * height * width + h * width + w;
                    out_data[idx] = (in_data[idx] - m) / s;
                }
            }
        }
    }

    return output;
}

// Apply augmentation pipeline
Tensor* augment_apply(Tensor* input, AugmentationConfig* config) {
    if (!input || !config)
        return NULL;

    Tensor* output = input;

    // Apply augmentations in order
    if (config->random_crop) {
        Tensor* cropped = augment_random_crop(output, config->crop_size[0], config->crop_size[1]);
        if (cropped && cropped != output) {
            if (output != input)
                tensor_free(output);
            output = cropped;
        }
    }

    if (config->random_horizontal_flip) {
        Tensor* flipped = augment_random_horizontal_flip(output, config->horizontal_flip_prob);
        if (flipped && flipped != output) {
            if (output != input)
                tensor_free(output);
            output = flipped;
        }
    }

    if (config->random_vertical_flip) {
        Tensor* flipped = augment_random_vertical_flip(output, config->vertical_flip_prob);
        if (flipped && flipped != output) {
            if (output != input)
                tensor_free(output);
            output = flipped;
        }
    }

    if (config->random_rotation) {
        Tensor* rotated =
            augment_random_rotation(output, config->rotation_angle_min, config->rotation_angle_max);
        if (rotated && rotated != output) {
            if (output != input)
                tensor_free(output);
            output = rotated;
        }
    }

    if (config->color_jitter) {
        Tensor* jittered = augment_color_jitter(output, config->brightness, config->contrast,
                                                config->saturation, config->hue);
        if (jittered && jittered != output) {
            if (output != input)
                tensor_free(output);
            output = jittered;
        }
    }

    if (config->normalize && config->mean && config->std) {
        Tensor* normalized =
            augment_normalize(output, config->mean, config->std, config->num_channels);
        if (normalized && normalized != output) {
            if (output != input)
                tensor_free(output);
            output = normalized;
        }
    }

    return output;
}
