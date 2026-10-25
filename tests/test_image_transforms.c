/* BMP I/O validated against an independently generated 2x2 BMP and a save->load round-trip,
 * plus flip / center-crop / bilinear-resize transforms checked against hand-computed pixels. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "alloc/cml_allocator.h"
#include "core/image_io.h"
#include "test_harness.h"

/* 2x2 BMP from Python struct: top-down (0,0)=red (0,1)=green (1,0)=blue (1,1)=white. */
static const unsigned char BMP_REF[] = {
    66, 77, 70, 0, 0,  0, 0,   0, 0, 0,   54,  0,   0, 0, 40, 0,  0,   0, 2,   0,  0, 0, 2, 0,
    0,  0,  1,  0, 24, 0, 0,   0, 0, 0,   16,  0,   0, 0, 19, 11, 0,   0, 19,  11, 0, 0, 0, 0,
    0,  0,  0,  0, 0,  0, 255, 0, 0, 255, 255, 255, 0, 0, 0,  0,  255, 0, 255, 0,  0, 0};

static float px_at(const float* im, int w, int c, int y, int x, int ch) {
    return im[((size_t)y * w + x) * c + ch];
}

static int test_load_bmp_ref(void) {
    FILE* f = fopen("/tmp/cml_test_ref.bmp", "wb");
    if (!f)
        return 0;
    fwrite(BMP_REF, 1, sizeof BMP_REF, f);
    fclose(f);
    int w = 0, h = 0, c = 0;
    float* im = cml_image_load_bmp("/tmp/cml_test_ref.bmp", &w, &h, &c);
    int ok    = im && w == 2 && h == 2 && c == 3;
    ok &= fabsf(px_at(im, 2, 3, 0, 0, 0) - 1.0f) < 1e-6f; /* (0,0) red */
    ok &= fabsf(px_at(im, 2, 3, 0, 1, 1) - 1.0f) < 1e-6f; /* (0,1) green */
    ok &= fabsf(px_at(im, 2, 3, 1, 0, 2) - 1.0f) < 1e-6f; /* (1,0) blue */
    ok &= fabsf(px_at(im, 2, 3, 1, 1, 0) - 1.0f) < 1e-6f; /* (1,1) white R */
    cml_free(im);
    return ok;
}

static int test_bmp_round_trip(void) {
    float img[12] = {0.2f, 0.4f, 0.6f, 0.0f, 1.0f, 0.5f,
                     1.0f, 0.0f, 0.0f, 0.3f, 0.3f, 0.3f}; /* 2x2 RGB */
    int ok        = cml_image_save_bmp("/tmp/cml_test_rt.bmp", img, 2, 2, 3) == 0;
    int w = 0, h = 0, c = 0;
    float* back = cml_image_load_bmp("/tmp/cml_test_rt.bmp", &w, &h, &c);
    ok &= back && w == 2 && h == 2 && c == 3;
    for (int i = 0; ok && i < 12; i++)
        ok &= fabsf(back[i] - img[i]) < 1.0f / 255.0f + 1e-4f; /* 8-bit quantization */
    cml_free(back);
    return ok;
}

static int test_flip(void) {
    float img[6] = {0.1f, 0.1f, 0.1f, 0.9f, 0.9f, 0.9f}; /* 2x1 */
    float* fl    = cml_image_flip_horizontal(img, 2, 1, 3);
    int ok       = fl && fabsf(fl[0] - 0.9f) < 1e-6f && fabsf(fl[3] - 0.1f) < 1e-6f;
    cml_free(fl);
    return ok;
}

static int test_center_crop(void) {
    float img[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8}; /* 3x3 single channel */
    int ow = 0, oh = 0;
    float* cr = cml_image_center_crop(img, 3, 3, 1, 1, 1, &ow, &oh);
    int ok    = cr && ow == 1 && oh == 1 && fabsf(cr[0] - 4.0f) < 1e-6f; /* center pixel */
    cml_free(cr);
    return ok;
}

static int test_resize(void) {
    float img[4] = {0.0f, 1.0f, 0.0f, 1.0f}; /* 2x2 single channel */
    float* up    = cml_image_resize_bilinear(img, 2, 2, 1, 4, 4);
    /* upsampling preserves the [0,1] range and corner values */
    int ok      = up && up[0] >= -1e-6f && up[15] <= 1.0f + 1e-6f;
    float* down = cml_image_resize_bilinear(img, 2, 2, 1, 1, 1); /* 1x1 average-ish */
    ok &= down && down[0] >= 0.0f && down[0] <= 1.0f;
    cml_free(up);
    cml_free(down);
    return ok;
}

int main(void) {
    printf("=== BMP I/O and image transforms ===\n");
    TEST(load_bmp_ref);
    TEST(bmp_round_trip);
    TEST(flip);
    TEST(center_crop);
    TEST(resize);
    return TEST_SUMMARY();
}
