#include "nn/layers/pixel_shuffle.h"
#include "nn.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

Tensor* f_pixel_shuffle(Tensor* input, int upscale_factor) {
    if (!input) {
        LOG_ERROR("f_pixel_shuffle: NULL input");
        return NULL;
    }
    if (input->ndim != 4) {
        LOG_ERROR("f_pixel_shuffle: expected 4D input [N, C*r^2, H, W], got %dD", input->ndim);
        return NULL;
    }

    int r = upscale_factor;
    if (r <= 0) {
        LOG_ERROR("f_pixel_shuffle: upscale_factor must be positive, got %d", r);
        return NULL;
    }

    int batch       = input->shape[0];
    int in_channels = input->shape[1];
    int in_h        = input->shape[2];
    int in_w        = input->shape[3];

    if (in_channels % (r * r) != 0) {
        LOG_ERROR("f_pixel_shuffle: input channels (%d) must be divisible by r^2 (%d)", in_channels,
                  r * r);
        return NULL;
    }

    int out_channels = in_channels / (r * r);

    /* Pure rearrange = reshape -> permute -> reshape (lazy, so it builds IR and
     * is graph-autodiff differentiable — was an eager ->data loop).
     * input[n, c*r*r+r1*r+r2, h, w] -> out[n, c, h*r+r1, w*r+r2]. */
    int s6[6]         = {batch, out_channels, r, r, in_h, in_w}; /* split C -> (c,r1,r2) */
    ReshapeParams rs1 = {s6, 6};
    Tensor* t1        = uop_reshape(input, &rs1);
    if (!t1)
        return NULL;

    int perm[6]      = {0, 1, 4, 2, 5, 3}; /* -> [N,C,H,r1,W,r2] */
    PermuteParams pp = {perm, 6};
    Tensor* t2       = uop_permute(t1, &pp);
    if (!t2)
        return NULL;

    int s4[4]         = {batch, out_channels, in_h * r, in_w * r};
    ReshapeParams rs2 = {s4, 4};
    return uop_reshape(t2, &rs2);
}

Tensor* f_pixel_unshuffle(Tensor* input, int downscale_factor) {
    if (!input) {
        LOG_ERROR("f_pixel_unshuffle: NULL input");
        return NULL;
    }

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
        LOG_ERROR("f_pixel_unshuffle: spatial dims (%d, %d) must be divisible by r (%d)", in_h,
                  in_w, r);
        return NULL;
    }

    int out_channels = in_channels * r * r;
    int out_h        = in_h / r;
    int out_w        = in_w / r;

    /* Inverse rearrange (lazy): reshape -> permute -> reshape.
     * input[n, c, h*r+r1, w*r+r2] -> out[n, c*r*r+r1*r+r2, h, w]. */
    int s6[6] = {batch, in_channels, out_h, r, out_w, r}; /* split H*r->(h,r1), W*r->(w,r2) */
    ReshapeParams rs1 = {s6, 6};
    Tensor* t1        = uop_reshape(input, &rs1);
    if (!t1)
        return NULL;

    int perm[6]      = {0, 1, 3, 5, 2, 4}; /* -> [N,C,r1,r2,h,w] */
    PermuteParams pp = {perm, 6};
    Tensor* t2       = uop_permute(t1, &pp);
    if (!t2)
        return NULL;

    int s4[4]         = {batch, out_channels, out_h, out_w};
    ReshapeParams rs2 = {s4, 4};
    return uop_reshape(t2, &rs2);
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
    cml_free(layer);
}

static void pixel_unshuffle_free(Module* module) {
    PixelUnshuffle* layer = (PixelUnshuffle*)module;
    if (!layer)
        return;
    cml_free(layer);
}

PixelShuffle* nn_pixel_shuffle(int upscale_factor) {
    if (upscale_factor <= 0) {
        LOG_ERROR("PixelShuffle: upscale_factor must be positive, got %d", upscale_factor);
        return NULL;
    }

    PixelShuffle* layer = cml_calloc(1, sizeof(PixelShuffle));
    if (!layer) {
        LOG_ERROR("PixelShuffle: failed to allocate memory");
        return NULL;
    }

    if (module_init((Module*)layer, "PixelShuffle", pixel_shuffle_forward, pixel_shuffle_free) !=
        0) {
        cml_free(layer);
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

    PixelUnshuffle* layer = cml_calloc(1, sizeof(PixelUnshuffle));
    if (!layer) {
        LOG_ERROR("PixelUnshuffle: failed to allocate memory");
        return NULL;
    }

    if (module_init((Module*)layer, "PixelUnshuffle", pixel_unshuffle_forward,
                    pixel_unshuffle_free) != 0) {
        cml_free(layer);
        return NULL;
    }

    layer->downscale_factor = downscale_factor;

    return layer;
}
