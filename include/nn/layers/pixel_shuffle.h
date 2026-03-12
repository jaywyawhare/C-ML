#ifndef CML_NN_LAYERS_PIXEL_SHUFFLE_H
#define CML_NN_LAYERS_PIXEL_SHUFFLE_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Rearranges [N, C*r^2, H, W] -> [N, C, H*r, W*r] */
typedef struct PixelShuffle {
    Module base;
    int upscale_factor;
} PixelShuffle;

/* Rearranges [N, C, H*r, W*r] -> [N, C*r^2, H, W] (inverse of PixelShuffle) */
typedef struct PixelUnshuffle {
    Module base;
    int downscale_factor;
} PixelUnshuffle;

PixelShuffle* nn_pixel_shuffle(int upscale_factor);
PixelUnshuffle* nn_pixel_unshuffle(int downscale_factor);

/* Functional versions (stateless, no module required) */
Tensor* f_pixel_shuffle(Tensor* input, int upscale_factor);
Tensor* f_pixel_unshuffle(Tensor* input, int downscale_factor);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_PIXEL_SHUFFLE_H
