#include "nn/layers/upsample.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float bicubic_kernel(float x) {
    float ax = fabsf(x);
    float a  = -0.5f;

    if (ax <= 1.0f) {
        return (a + 2.0f) * ax * ax * ax - (a + 3.0f) * ax * ax + 1.0f;
    } else if (ax < 2.0f) {
        return a * ax * ax * ax - 5.0f * a * ax * ax + 8.0f * a * ax - 4.0f * a;
    }
    return 0.0f;
}

static Tensor* interpolate_nearest_4d(Tensor* input, int out_h, int out_w) {
    int batch      = input->shape[0];
    int channels   = input->shape[1];
    int in_h       = input->shape[2];
    int in_w       = input->shape[3];

    int out_shape[] = {batch, channels, out_h, out_w};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 4, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)input->data;
    float* out_data = (float*)output->data;

    float scale_h = (float)in_h / (float)out_h;
    float scale_w = (float)in_w / (float)out_w;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                int src_h = (int)floorf(oh * scale_h);
                if (src_h >= in_h) src_h = in_h - 1;

                for (int ow = 0; ow < out_w; ow++) {
                    int src_w = (int)floorf(ow * scale_w);
                    if (src_w >= in_w) src_w = in_w - 1;

                    out_data[((b * channels + c) * out_h + oh) * out_w + ow] =
                        in_data[((b * channels + c) * in_h + src_h) * in_w + src_w];
                }
            }
        }
    }

    return output;
}

static Tensor* interpolate_bilinear_4d(Tensor* input, int out_h, int out_w,
                                        bool align_corners) {
    int batch    = input->shape[0];
    int channels = input->shape[1];
    int in_h     = input->shape[2];
    int in_w     = input->shape[3];

    int out_shape[] = {batch, channels, out_h, out_w};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_zeros(out_shape, 4, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)input->data;
    float* out_data = (float*)output->data;

    float scale_h, scale_w;
    if (align_corners && out_h > 1) {
        scale_h = (float)(in_h - 1) / (float)(out_h - 1);
    } else {
        scale_h = (float)in_h / (float)out_h;
    }
    if (align_corners && out_w > 1) {
        scale_w = (float)(in_w - 1) / (float)(out_w - 1);
    } else {
        scale_w = (float)in_w / (float)out_w;
    }

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                float src_h;
                if (align_corners && out_h > 1) {
                    src_h = oh * scale_h;
                } else {
                    src_h = (oh + 0.5f) * scale_h - 0.5f;
                }

                int h0 = (int)floorf(src_h);
                int h1 = h0 + 1;
                float wh = src_h - h0;

                if (h0 < 0) h0 = 0;
                if (h1 >= in_h) h1 = in_h - 1;

                for (int ow = 0; ow < out_w; ow++) {
                    float src_w;
                    if (align_corners && out_w > 1) {
                        src_w = ow * scale_w;
                    } else {
                        src_w = (ow + 0.5f) * scale_w - 0.5f;
                    }

                    int w0 = (int)floorf(src_w);
                    int w1 = w0 + 1;
                    float ww = src_w - w0;

                    if (w0 < 0) w0 = 0;
                    if (w1 >= in_w) w1 = in_w - 1;

                    int base = (b * channels + c) * in_h;
                    float v00 = in_data[(base + h0) * in_w + w0];
                    float v01 = in_data[(base + h0) * in_w + w1];
                    float v10 = in_data[(base + h1) * in_w + w0];
                    float v11 = in_data[(base + h1) * in_w + w1];

                    float val = v00 * (1 - wh) * (1 - ww) +
                                v01 * (1 - wh) * ww +
                                v10 * wh * (1 - ww) +
                                v11 * wh * ww;

                    out_data[((b * channels + c) * out_h + oh) * out_w + ow] = val;
                }
            }
        }
    }

    return output;
}

static inline int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static Tensor* interpolate_bicubic_4d(Tensor* input, int out_h, int out_w,
                                       bool align_corners) {
    int batch    = input->shape[0];
    int channels = input->shape[1];
    int in_h     = input->shape[2];
    int in_w     = input->shape[3];

    int out_shape[] = {batch, channels, out_h, out_w};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_zeros(out_shape, 4, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)input->data;
    float* out_data = (float*)output->data;

    float scale_h, scale_w;
    if (align_corners && out_h > 1) {
        scale_h = (float)(in_h - 1) / (float)(out_h - 1);
    } else {
        scale_h = (float)in_h / (float)out_h;
    }
    if (align_corners && out_w > 1) {
        scale_w = (float)(in_w - 1) / (float)(out_w - 1);
    } else {
        scale_w = (float)in_w / (float)out_w;
    }

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            int base = (b * channels + c) * in_h;

            for (int oh = 0; oh < out_h; oh++) {
                float src_h;
                if (align_corners && out_h > 1) {
                    src_h = oh * scale_h;
                } else {
                    src_h = (oh + 0.5f) * scale_h - 0.5f;
                }

                int h_floor = (int)floorf(src_h);
                float fh    = src_h - h_floor;

                for (int ow = 0; ow < out_w; ow++) {
                    float src_w;
                    if (align_corners && out_w > 1) {
                        src_w = ow * scale_w;
                    } else {
                        src_w = (ow + 0.5f) * scale_w - 0.5f;
                    }

                    int w_floor = (int)floorf(src_w);
                    float fw    = src_w - w_floor;

                    float val = 0.0f;

                    /* 4x4 bicubic neighbourhood */
                    for (int j = -1; j <= 2; j++) {
                        float ky = bicubic_kernel(fh - j);
                        int sy   = clamp_int(h_floor + j, 0, in_h - 1);

                        for (int i = -1; i <= 2; i++) {
                            float kx = bicubic_kernel(fw - i);
                            int sx   = clamp_int(w_floor + i, 0, in_w - 1);

                            val += ky * kx * in_data[(base + sy) * in_w + sx];
                        }
                    }

                    out_data[((b * channels + c) * out_h + oh) * out_w + ow] = val;
                }
            }
        }
    }

    return output;
}

Tensor* f_interpolate(Tensor* input, const int* output_size, int num_dims,
                      UpsampleMode mode, bool align_corners) {
    if (!input) {
        LOG_ERROR("f_interpolate: NULL input");
        return NULL;
    }

    tensor_ensure_executed(input);

    if (input->ndim != 4) {
        LOG_ERROR("f_interpolate: expected 4D input [N,C,H,W], got %dD", input->ndim);
        return NULL;
    }

    if (!output_size || num_dims < 2) {
        LOG_ERROR("f_interpolate: output_size must have at least 2 elements");
        return NULL;
    }

    int out_h = output_size[0];
    int out_w = output_size[1];

    if (out_h <= 0 || out_w <= 0) {
        LOG_ERROR("f_interpolate: invalid output size (%d, %d)", out_h, out_w);
        return NULL;
    }

    switch (mode) {
        case UPSAMPLE_NEAREST:
            return interpolate_nearest_4d(input, out_h, out_w);
        case UPSAMPLE_BILINEAR:
            return interpolate_bilinear_4d(input, out_h, out_w, align_corners);
        case UPSAMPLE_BICUBIC:
            return interpolate_bicubic_4d(input, out_h, out_w, align_corners);
        default:
            LOG_ERROR("f_interpolate: unsupported mode %d", (int)mode);
            return NULL;
    }
}

Tensor* upsample_forward(Module* module, Tensor* input) {
    Upsample* layer = (Upsample*)module;

    if (!layer || !input)
        return NULL;

    tensor_ensure_executed(input);

    if (input->ndim != 4) {
        LOG_ERROR("Upsample forward: expected 4D input [N,C,H,W], got %dD", input->ndim);
        return NULL;
    }

    int in_h = input->shape[2];
    int in_w = input->shape[3];

    int out_h, out_w;

    if (layer->scale_factor > 0.0f) {
        out_h = (int)(in_h * layer->scale_factor);
        out_w = (int)(in_w * layer->scale_factor);
    } else if (layer->num_output_dims >= 2) {
        out_h = layer->output_size[0];
        out_w = layer->output_size[1];
    } else {
        LOG_ERROR("Upsample: neither scale_factor nor output_size is set");
        return NULL;
    }

    if (out_h <= 0 || out_w <= 0) {
        LOG_ERROR("Upsample: invalid computed output size (%d, %d)", out_h, out_w);
        return NULL;
    }

    /* For nearest and bilinear, delegate to tensor_interpolate when the mode
     * maps directly.  Bicubic is handled locally. */
    if (layer->mode == UPSAMPLE_NEAREST) {
        int size[] = {out_h, out_w};
        Tensor* result = tensor_interpolate(input, size, 2, INTERP_NEAREST);
        return result;
    } else if (layer->mode == UPSAMPLE_BILINEAR) {
        int size[] = {out_h, out_w};
        Tensor* result = tensor_interpolate(input, size, 2, INTERP_BILINEAR);
        return result;
    } else if (layer->mode == UPSAMPLE_BICUBIC) {
        return interpolate_bicubic_4d(input, out_h, out_w, layer->align_corners);
    }

    LOG_ERROR("Upsample: unsupported mode %d", (int)layer->mode);
    return NULL;
}

static void upsample_free(Module* module) {
    Upsample* layer = (Upsample*)module;
    if (!layer)
        return;
    free(layer);
}

Upsample* nn_upsample(float scale_factor, const int* output_size, int num_output_dims,
                       UpsampleMode mode, bool align_corners) {
    Upsample* layer = calloc(1, sizeof(Upsample));
    if (!layer) {
        LOG_ERROR("Upsample: failed to allocate memory");
        return NULL;
    }

    if (module_init((Module*)layer, "Upsample", upsample_forward, upsample_free) != 0) {
        free(layer);
        return NULL;
    }

    layer->scale_factor    = scale_factor;
    layer->num_output_dims = 0;
    layer->mode            = mode;
    layer->align_corners   = align_corners;

    if (scale_factor <= 0.0f && output_size && num_output_dims > 0) {
        int n = num_output_dims;
        if (n > UPSAMPLE_MAX_DIMS) n = UPSAMPLE_MAX_DIMS;
        for (int i = 0; i < n; i++) {
            layer->output_size[i] = output_size[i];
        }
        layer->num_output_dims = n;
    }

    LOG_DEBUG("Created Upsample layer: mode=%d, scale=%.2f, align_corners=%d",
              (int)mode, scale_factor, (int)align_corners);

    return layer;
}
