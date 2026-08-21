/* Correctness test for augmentation.c (previously untested): per-channel
 * normalize, deterministic horizontal flip (prob=1), and crop output shape. */
#include <stdio.h>
#include <math.h>
#include "cml.h"
#include "core/augmentation.h"
#include "test_harness.h"


int main(void) {
    cml_init();
    printf("=== augmentation ===\n");

    /* normalize: [B=1, C=2, H=2, W=2], (x-mean[c])/std[c] per channel */
    { float d[8] = { 1,3,5,7,   2,4,6,8 };   /* ch0: 1..7, ch1: 2..8 */
      int shp[4] = {1,2,2,2};
      Tensor* x = cml_tensor(d, shp, 4, NULL);
      float mean[2] = {1.0f, 0.0f}, std[2] = {2.0f, 1.0f};
      Tensor* y = augment_normalize(x, mean, std, 2);
      float* o = y ? (float*)tensor_data_ptr(y) : NULL;
      /* ch0: (d-1)/2 -> 0,1,2,3 ; ch1: (d-0)/1 -> 2,4,6,8 */
      float want[8] = {0,1,2,3, 2,4,6,8};
      int ok = (o != NULL);
      for (int i = 0; i < 8 && ok; i++) if (fabsf(o[i] - want[i]) > 1e-5f) ok = 0;
      CHECK("augment_normalize per-channel correct", ok); }

    /* horizontal flip with prob=1.0 reverses along width (last dim) */
    { float d[8] = { 1,2, 3,4,   5,6, 7,8 };  /* [1,2,2,2], each row [a,b] -> [b,a] */
      int shp[4] = {1,2,2,2};
      Tensor* x = cml_tensor(d, shp, 4, NULL);
      Tensor* y = augment_random_horizontal_flip(x, 1.0f);
      float* o = y ? (float*)tensor_data_ptr(y) : NULL;
      float want[8] = { 2,1, 4,3,   6,5, 8,7 };
      int ok = (o != NULL);
      for (int i = 0; i < 8 && ok; i++) if (fabsf(o[i] - want[i]) > 1e-5f) ok = 0;
      CHECK("horizontal flip (prob=1) reverses width", ok); }

    /* flip with prob=0.0 is identity */
    { float d[4] = {1,2,3,4}; int shp[4] = {1,1,2,2};
      Tensor* x = cml_tensor(d, shp, 4, NULL);
      Tensor* y = augment_random_horizontal_flip(x, 0.0f);
      float* o = y ? (float*)tensor_data_ptr(y) : NULL;
      int ok = o && fabsf(o[0]-1) < 1e-5f && fabsf(o[1]-2) < 1e-5f && fabsf(o[3]-4) < 1e-5f;
      CHECK("horizontal flip (prob=0) is identity", ok); }

    /* random crop produces the requested output size */
    { float d[16]; for (int i = 0; i < 16; i++) d[i] = (float)i;
      int shp[4] = {1,1,4,4};
      Tensor* x = cml_tensor(d, shp, 4, NULL);
      Tensor* y = augment_random_crop(x, 2, 2);
      CHECK("random crop output shape [1,1,2,2]",
            y && y->ndim == 4 && y->shape[2] == 2 && y->shape[3] == 2); }

    return TEST_SUMMARY();
}
