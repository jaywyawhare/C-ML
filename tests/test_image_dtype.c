#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor/image_dtype.h"
#include "tensor/tensor.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

static int test_channels_rgba32f(void) {
    return cml_image_dtype_channels(CML_IMAGE_RGBA32F) == 4;
}

static int test_channels_rg32f(void) {
    return cml_image_dtype_channels(CML_IMAGE_RG32F) == 2;
}

static int test_channels_r32f(void) {
    return cml_image_dtype_channels(CML_IMAGE_R32F) == 1;
}

static int test_bpp_rgba32f(void) {
    return cml_image_dtype_bpp(CML_IMAGE_RGBA32F) == 16;
}

static int test_bpp_rgba8(void) {
    return cml_image_dtype_bpp(CML_IMAGE_RGBA8) == 4;
}

static int test_bpp_r16f(void) {
    return cml_image_dtype_bpp(CML_IMAGE_R16F) == 2;
}

static int test_name(void) {
    return strcmp(cml_image_dtype_name(CML_IMAGE_RGBA32F), "rgba32f") == 0;
}

static int test_compatible_rgba(void) {
    int shape[] = {16, 16, 4};
    return cml_image_dtype_compatible(shape, 3, CML_IMAGE_RGBA32F);
}

static int test_incompatible_channels(void) {
    int shape[] = {16, 16, 3};
    return !cml_image_dtype_compatible(shape, 3, CML_IMAGE_RGBA32F);
}

static int test_select_format_div4(void) {
    int shape[] = {8, 8, 4};
    return cml_image_dtype_select(shape, 3) == CML_IMAGE_RGBA32F;
}

static int test_select_format_div2(void) {
    int shape[] = {8, 6};
    return cml_image_dtype_select(shape, 2) == CML_IMAGE_RG32F;
}

static int test_select_format_odd(void) {
    int shape[] = {8, 3};
    return cml_image_dtype_select(shape, 2) == CML_IMAGE_R32F;
}

static int test_image_dims(void) {
    int shape[] = {4, 4};
    int w, h;
    cml_image_dtype_dims(shape, 2, CML_IMAGE_R32F, &w, &h);
    return (w * h == 16);
}

static int test_create_image_tensor(void) {
    int shape[] = {4, 4};
    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, 2, &tc);
    if (!t) return 0;
    CMLImageTensor* img = cml_image_tensor_create(t, CML_IMAGE_R32F);
    int ok = (img != NULL);
    if (ok) {
        ok = ok && (img->format == CML_IMAGE_R32F);
        ok = ok && (img->channels == 1);
        cml_image_tensor_free(img);
    }
    tensor_free(t);
    return ok;
}

static int test_create_incompatible(void) {
    int shape[] = {4, 3};
    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, 2, &tc);
    if (!t) return 0;
    CMLImageTensor* img = cml_image_tensor_create(t, CML_IMAGE_RGBA32F);
    int ok = (img == NULL); /* Should fail: 3 not divisible by 4 */
    tensor_free(t);
    return ok;
}

static int test_to_regular(void) {
    int shape[] = {4, 4};
    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, 2, &tc);
    if (!t) return 0;
    CMLImageTensor* img = cml_image_tensor_create(t, CML_IMAGE_R32F);
    if (!img) { tensor_free(t); return 0; }
    Tensor* regular = cml_image_tensor_to_regular(img);
    int ok = (regular == t); /* Should return same tensor */
    cml_image_tensor_free(img);
    tensor_free(t);
    return ok;
}

static int test_memory_size(void) {
    int shape[] = {8, 4};
    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, 2, &tc);
    if (!t) return 0;
    CMLImageTensor* img = cml_image_tensor_create(t, CML_IMAGE_RGBA32F);
    if (!img) { tensor_free(t); return 0; }
    size_t mem = cml_image_tensor_memory(img);
    int ok = (mem > 0);
    cml_image_tensor_free(img);
    tensor_free(t);
    return ok;
}

static int test_print_no_crash(void) {
    int shape[] = {4, 4};
    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, 2, &tc);
    if (!t) return 0;
    CMLImageTensor* img = cml_image_tensor_create(t, CML_IMAGE_R32F);
    cml_image_tensor_print(img);
    cml_image_tensor_print(NULL);
    if (img) cml_image_tensor_free(img);
    tensor_free(t);
    return 1;
}

static int test_free_null(void) {
    cml_image_tensor_free(NULL);
    return 1;
}

int main(void) {
    printf("Image Dtype Tests\n");

    RUN_TEST(test_channels_rgba32f);
    RUN_TEST(test_channels_rg32f);
    RUN_TEST(test_channels_r32f);
    RUN_TEST(test_bpp_rgba32f);
    RUN_TEST(test_bpp_rgba8);
    RUN_TEST(test_bpp_r16f);
    RUN_TEST(test_name);
    RUN_TEST(test_compatible_rgba);
    RUN_TEST(test_incompatible_channels);
    RUN_TEST(test_select_format_div4);
    RUN_TEST(test_select_format_div2);
    RUN_TEST(test_select_format_odd);
    RUN_TEST(test_image_dims);
    RUN_TEST(test_create_image_tensor);
    RUN_TEST(test_create_incompatible);
    RUN_TEST(test_to_regular);
    RUN_TEST(test_memory_size);
    RUN_TEST(test_print_no_crash);
    RUN_TEST(test_free_null);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
