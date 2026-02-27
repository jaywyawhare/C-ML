/**
 * @file augmentation.c
 * @brief Data augmentation implementation
 */

#include "core/augmentation.h"
#include "core/logging.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

AugmentationConfig* augmentation_config_create(void) {
    AugmentationConfig* config = malloc(sizeof(AugmentationConfig));
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
        free(config->mean);
    if (config->std)
        free(config->std);
    free(config);
}

// Random number generator (simple linear congruential)
static unsigned int rng_seed = 1;
static void set_seed(unsigned int seed) { rng_seed = seed; }
static float rand_float(void) {
    rng_seed = (rng_seed * 1103515245 + 12345) & 0x7fffffff;
    return (float)rng_seed / 2147483648.0f;
}

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
    int h_offset = (int)(rand_float() * (float)(height - crop_height));
    int w_offset = (int)(rand_float() * (float)(width - crop_width));

    // Create output tensor
    int output_shape[]  = {batch, channels, crop_height, crop_width};
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(output_shape, 4, &config);
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
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(input->shape, 4, &config);
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
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(input->shape, 4, &config);
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

    // Generate random angle in radians
    float angle_deg = angle_min + rand_float() * (angle_max - angle_min);
    float angle_rad = angle_deg * (float)M_PI / 180.0f;

    // Precompute rotation matrix elements
    float cos_a = cosf(angle_rad);
    float sin_a = sinf(angle_rad);

    // Check for 90-degree multiples for optimization
    int rotation_90     = (int)(angle_deg / 90.0f + 0.5f) % 4;
    bool is_90_multiple = fabsf(angle_deg - (float)rotation_90 * 90.0f) < 0.1f;

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    // Create output tensor
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(input->shape, 4, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);

    // Center of rotation
    float center_x = (float)(width - 1) / 2.0f;
    float center_y = (float)(height - 1) / 2.0f;

    // Apply rotation with bilinear interpolation
    if (is_90_multiple) {
        // Optimized path for 90-degree multiples
        int rotation = rotation_90;
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
    } else {
        // Arbitrary angle rotation with bilinear interpolation
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int out_h = 0; out_h < height; out_h++) {
                    for (int out_w = 0; out_w < width; out_w++) {
                        // Transform output coordinates to input coordinates (inverse rotation)
                        float x = (float)out_w - center_x;
                        float y = (float)out_h - center_y;

                        // Apply inverse rotation
                        float in_x = x * cos_a + y * sin_a + center_x;
                        float in_y = -x * sin_a + y * cos_a + center_y;

                        // Bilinear interpolation
                        int x0 = (int)floorf(in_x);
                        int y0 = (int)floorf(in_y);
                        int x1 = x0 + 1;
                        int y1 = y0 + 1;

                        float dx = in_x - (float)x0;
                        float dy = in_y - (float)y0;

                        // Clamp coordinates
                        x0 = (x0 < 0) ? 0 : (x0 >= width ? width - 1 : x0);
                        y0 = (y0 < 0) ? 0 : (y0 >= height ? height - 1 : y0);
                        x1 = (x1 < 0) ? 0 : (x1 >= width ? width - 1 : x1);
                        y1 = (y1 < 0) ? 0 : (y1 >= height ? height - 1 : y1);

                        // Get pixel values
                        int idx00 =
                            b * channels * height * width + c * height * width + y0 * width + x0;
                        int idx01 =
                            b * channels * height * width + c * height * width + y0 * width + x1;
                        int idx10 =
                            b * channels * height * width + c * height * width + y1 * width + x0;
                        int idx11 =
                            b * channels * height * width + c * height * width + y1 * width + x1;

                        // Bilinear interpolation
                        float val = in_data[idx00] * (1.0f - dx) * (1.0f - dy) +
                                    in_data[idx01] * dx * (1.0f - dy) +
                                    in_data[idx10] * (1.0f - dx) * dy + in_data[idx11] * dx * dy;

                        int out_idx = b * channels * height * width + c * height * width +
                                      out_h * width + out_w;
                        out_data[out_idx] = val;
                    }
                }
            }
        }
    }

    return output;
}

// Color Jitter (brightness, contrast, saturation, hue)
Tensor* augment_color_jitter(Tensor* input, float brightness, float contrast, float saturation,
                             float hue) {
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
    float sat_factor      = 1.0f + (rand_float() * 2.0f - 1.0f) * saturation;
    float hue_factor      = (rand_float() * 2.0f - 1.0f) * hue; // Hue shift in [-hue, +hue]

    // Create output tensor
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(input->shape, 4, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);

    int batch    = input->shape[0];
    int channels = input->shape[1];
    int height   = input->shape[2];
    int width    = input->shape[3];

    // Apply jitter
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // Get RGB values
                float r =
                    in_data[b * channels * height * width + 0 * height * width + h * width + w];
                float g     = channels > 1 ? in_data[b * channels * height * width +
                                                 1 * height * width + h * width + w]
                                           : r;
                float b_val = channels > 2 ? in_data[b * channels * height * width +
                                                     2 * height * width + h * width + w]
                                           : r;

                // Apply brightness
                r *= bright_factor;
                g *= bright_factor;
                b_val *= bright_factor;

                // Apply contrast
                r     = (r - 0.5f) * contrast_factor + 0.5f;
                g     = (g - 0.5f) * contrast_factor + 0.5f;
                b_val = (b_val - 0.5f) * contrast_factor + 0.5f;

                // Apply saturation and hue (convert RGB to HSV, modify, convert back)
                if (channels >= 3 &&
                    (fabsf(sat_factor - 1.0f) > 1e-6f || fabsf(hue_factor) > 1e-6f)) {
                    // Convert RGB to HSV
                    float max_val = r > g ? (r > b_val ? r : b_val) : (g > b_val ? g : b_val);
                    float min_val = r < g ? (r < b_val ? r : b_val) : (g < b_val ? g : b_val);
                    float delta   = max_val - min_val;

                    float v     = max_val;
                    float s_val = (max_val > 0.0f) ? delta / max_val : 0.0f;
                    float h_val = 0.0f;

                    if (delta > 0.0f) {
                        if (fabsf(max_val - r) < 1e-6f) {
                            h_val = 60.0f * (((g - b_val) / delta));
                        } else if (fabsf(max_val - g) < 1e-6f) {
                            h_val = 60.0f * (((b_val - r) / delta) + 2.0f);
                        } else {
                            h_val = 60.0f * (((r - g) / delta) + 4.0f);
                        }
                        if (h_val < 0.0f)
                            h_val += 360.0f;
                    }

                    // Apply saturation and hue adjustments
                    s_val *= sat_factor;
                    if (s_val > 1.0f)
                        s_val = 1.0f;
                    if (s_val < 0.0f)
                        s_val = 0.0f;

                    h_val += hue_factor * 360.0f;
                    while (h_val < 0.0f)
                        h_val += 360.0f;
                    while (h_val >= 360.0f)
                        h_val -= 360.0f;

                    // Convert HSV back to RGB
                    float c = v * s_val;
                    float x = c * (1.0f - fabsf(fmodf(h_val / 60.0f, 2.0f) - 1.0f));
                    float m = v - c;

                    if (h_val < 60.0f) {
                        r     = c + m;
                        g     = x + m;
                        b_val = m;
                    } else if (h_val < 120.0f) {
                        r     = x + m;
                        g     = c + m;
                        b_val = m;
                    } else if (h_val < 180.0f) {
                        r     = m;
                        g     = c + m;
                        b_val = x + m;
                    } else if (h_val < 240.0f) {
                        r     = m;
                        g     = x + m;
                        b_val = c + m;
                    } else if (h_val < 300.0f) {
                        r     = x + m;
                        g     = m;
                        b_val = c + m;
                    } else {
                        r     = c + m;
                        g     = m;
                        b_val = x + m;
                    }
                }

                // Clamp to [0, 1]
                r     = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
                g     = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
                b_val = b_val < 0.0f ? 0.0f : (b_val > 1.0f ? 1.0f : b_val);

                // Write back
                out_data[b * channels * height * width + 0 * height * width + h * width + w] = r;
                if (channels > 1) {
                    out_data[b * channels * height * width + 1 * height * width + h * width + w] =
                        g;
                }
                if (channels > 2) {
                    out_data[b * channels * height * width + 2 * height * width + h * width + w] =
                        b_val;
                }
            }
        }
    }

    return output;
}

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
    TensorConfig config = {
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(input->shape, 4, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)tensor_data_ptr(input);
    float* out_data = (float*)tensor_data_ptr(output);

    // Normalize: (x - mean) / std
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            float m = mean[c];
            float s = std[c];
            if (fabsf(s) < 1e-6f)
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
