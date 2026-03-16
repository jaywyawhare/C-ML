#include "nn/layers/pixel_shuffle.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

Tensor* f_pixel_shuffle(Tensor* input, int upscale_factor) {
    if (!input) {
        LOG_ERROR("f_pixel_shuffle: NULL input");
        return NULL;
    }

    tensor_ensure_executed(input);

    if (input->ndim != 4) {
        LOG_ERROR("f_pixel_shuffle: expected 4D input [N, C*r^2, H, W], got %dD", input->ndim);
        return NULL;
    }

    int r = upscale_factor;
    if (r <= 0) {
        LOG_ERROR("f_pixel_shuffle: upscale_factor must be positive, got %d", r);
        return NULL;
    }

    int batch      = input->shape[0];
    int in_channels = input->shape[1];
    int in_h       = input->shape[2];
    int in_w       = input->shape[3];

    if (in_channels % (r * r) != 0) {
        LOG_ERROR("f_pixel_shuffle: input channels (%d) must be divisible by r^2 (%d)",
                  in_channels, r * r);
        return NULL;
    }

    int out_channels = in_channels / (r * r);
    int out_h        = in_h * r;
    int out_w        = in_w * r;

    int out_shape[] = {batch, out_channels, out_h, out_w};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 4, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)input->data;
    float* out_data = (float*)output->data;

    if (!in_data || !out_data) {
        tensor_free(output);
        return NULL;
    }

    /* Rearrange: input[n, c*r*r + r1*r + r2, h, w] -> output[n, c, h*r + r1, w*r + r2] */
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < out_channels; c++) {
            for (int r1 = 0; r1 < r; r1++) {
                for (int r2 = 0; r2 < r; r2++) {
                    int ic = c * r * r + r1 * r + r2;

                    for (int h = 0; h < in_h; h++) {
                        for (int w = 0; w < in_w; w++) {
                            int oh = h * r + r1;
                            int ow = w * r + r2;

                            float val = in_data[((n * in_channels + ic) * in_h + h) * in_w + w];
                            out_data[((n * out_channels + c) * out_h + oh) * out_w + ow] = val;
                        }
                    }
                }
            }
        }
    }

    return output;
}

Tensor* f_pixel_unshuffle(Tensor* input, int downscale_factor) {
    if (!input) {
        LOG_ERROR("f_pixel_unshuffle: NULL input");
        return NULL;
    }

    tensor_ensure_executed(input);

    if (input->ndim != 4) {
        LOG_ERROR("f_pixel_unshuffle: expected 4D input [N, C, H*r, W*r], got %dD", input->ndim);
        return NULL;
    }

    int r = downscale_factor;
    if (r <= 0) {
        LOG_ERROR("f_pixel_unshuffle: downscale_factor must be positive, got %d", r);
        return NULL;
    }

    int batch       = input->shape[0];
    int in_channels = input->shape[1];
    int in_h        = input->shape[2];
    int in_w        = input->shape[3];

    if (in_h % r != 0 || in_w % r != 0) {
        LOG_ERROR("f_pixel_unshuffle: spatial dims (%d, %d) must be divisible by r (%d)",
                  in_h, in_w, r);
        return NULL;
    }

    int out_channels = in_channels * r * r;
    int out_h        = in_h / r;
    int out_w        = in_w / r;

    int out_shape[] = {batch, out_channels, out_h, out_w};
    TensorConfig config = (TensorConfig){
        .dtype = input->dtype, .device = input->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, 4, &config);
    if (!output)
        return NULL;

    float* in_data  = (float*)input->data;
    float* out_data = (float*)output->data;

    if (!in_data || !out_data) {
        tensor_free(output);
        return NULL;
    }

    /* Rearrange: input[n, c, h*r + r1, w*r + r2] -> output[n, c*r*r + r1*r + r2, h, w] */
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < in_channels; c++) {
            for (int r1 = 0; r1 < r; r1++) {
                for (int r2 = 0; r2 < r; r2++) {
                    int oc = c * r * r + r1 * r + r2;

                    for (int h = 0; h < out_h; h++) {
                        for (int w = 0; w < out_w; w++) {
                            int ih = h * r + r1;
                            int iw = w * r + r2;

                            float val = in_data[((n * in_channels + c) * in_h + ih) * in_w + iw];
                            out_data[((n * out_channels + oc) * out_h + h) * out_w + w] = val;
                        }
                    }
                }
            }
        }
    }

    return output;
}

static Tensor* pixel_shuffle_forward(Module* module, Tensor* input) {
    PixelShuffle* layer = (PixelShuffle*)module;
    if (!layer || !input)
        return NULL;
    return f_pixel_shuffle(input, layer->upscale_factor);
}

static Tensor* pixel_unshuffle_forward(Module* module, Tensor* input) {
    PixelUnshuffle* layer = (PixelUnshuffle*)module;
    if (!layer || !input)
        return NULL;
    return f_pixel_unshuffle(input, layer->downscale_factor);
}

static void pixel_shuffle_free(Module* module) {
    PixelShuffle* layer = (PixelShuffle*)module;
    if (!layer)
        return;
    free(layer);
}

static void pixel_unshuffle_free(Module* module) {
    PixelUnshuffle* layer = (PixelUnshuffle*)module;
    if (!layer)
        return;
    free(layer);
}

PixelShuffle* nn_pixel_shuffle(int upscale_factor) {
    if (upscale_factor <= 0) {
        LOG_ERROR("PixelShuffle: upscale_factor must be positive, got %d", upscale_factor);
        return NULL;
    }

    PixelShuffle* layer = calloc(1, sizeof(PixelShuffle));
    if (!layer) {
        LOG_ERROR("PixelShuffle: failed to allocate memory");
        return NULL;
    }

    if (module_init((Module*)layer, "PixelShuffle", pixel_shuffle_forward,
                    pixel_shuffle_free) != 0) {
        free(layer);
        return NULL;
    }

    layer->upscale_factor = upscale_factor;

    return layer;
}

PixelUnshuffle* nn_pixel_unshuffle(int downscale_factor) {
    if (downscale_factor <= 0) {
        LOG_ERROR("PixelUnshuffle: downscale_factor must be positive, got %d", downscale_factor);
        return NULL;
    }

    PixelUnshuffle* layer = calloc(1, sizeof(PixelUnshuffle));
    if (!layer) {
        LOG_ERROR("PixelUnshuffle: failed to allocate memory");
        return NULL;
    }

    if (module_init((Module*)layer, "PixelUnshuffle", pixel_unshuffle_forward,
                    pixel_unshuffle_free) != 0) {
        free(layer);
        return NULL;
    }

    layer->downscale_factor = downscale_factor;

    return layer;
}
