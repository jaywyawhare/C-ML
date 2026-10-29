#ifndef CML_NN_GRID_SAMPLE_H
#define CML_NN_GRID_SAMPLE_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { CML_GRID_BILINEAR = 0, CML_GRID_NEAREST = 1 } CMLGridSampleMode;

typedef enum { CML_GRID_PAD_ZEROS = 0, CML_GRID_PAD_BORDER = 1 } CMLGridPadMode;

/**
 * Build a 2-D sampling grid from a batch of affine matrices (as torch's
 * affine_grid). Differentiable in @p theta via matmul.
 * @param theta (Tensor*) affine matrices [N, 2, 3]
 * @param N (int) batch size
 * @param H (int) output grid height
 * @param W (int) output grid width
 * @param align_corners (bool) corner-aligned normalization if true
 * @return (Tensor*) sampling grid [N, H, W, 2] in normalized [-1,1] coords, or NULL
 */
Tensor* cml_affine_grid(Tensor* theta, int N, int H, int W, bool align_corners);

/**
 * Sample an image batch at normalized grid coordinates (forward pass of torch's
 * grid_sample). Operates on realized f32 buffers.
 * @param input (Tensor*) [N, C, H_in, W_in]
 * @param grid (Tensor*) [N, H_out, W_out, 2] normalized coords, last dim (x, y)
 * @param mode (CMLGridSampleMode) bilinear or nearest
 * @param padding (CMLGridPadMode) out-of-bounds handling
 * @param align_corners (bool) corner-aligned coordinate mapping if true
 * @return (Tensor*) sampled output [N, C, H_out, W_out], or NULL on error
 */
Tensor* cml_grid_sample(Tensor* input, Tensor* grid, CMLGridSampleMode mode, CMLGridPadMode padding,
                        bool align_corners);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_GRID_SAMPLE_H */
