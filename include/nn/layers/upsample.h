#ifndef CML_NN_LAYERS_UPSAMPLE_H
#define CML_NN_LAYERS_UPSAMPLE_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    UPSAMPLE_NEAREST,
    UPSAMPLE_BILINEAR,
    UPSAMPLE_BICUBIC
} UpsampleMode;

#define UPSAMPLE_MAX_DIMS 3

typedef struct Upsample {
    Module base;

    float scale_factor;                    /* 0 if using output_size instead */
    int output_size[UPSAMPLE_MAX_DIMS];   /* 0 if using scale_factor instead */
    int num_output_dims;
    UpsampleMode mode;
    bool align_corners;
} Upsample;

/*
 * One of scale_factor or output_size must be specified.
 * If scale_factor > 0, output_size is ignored.
 */
Upsample* nn_upsample(float scale_factor, const int* output_size, int num_output_dims,
                       UpsampleMode mode, bool align_corners);

Tensor* upsample_forward(Module* module, Tensor* input);

/* Functional interpolation (stateless, no module required) */
Tensor* f_interpolate(Tensor* input, const int* output_size, int num_dims,
                      UpsampleMode mode, bool align_corners);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_UPSAMPLE_H
