/**
 * @file test_winograd.c
 * @brief Tests for Winograd convolution transforms
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "ops/winograd.h"

static void test_winograd_applicable_true(void) {
    printf("  test_winograd_applicable_true...");
    /* 3x3 kernel, stride 1, dilation 1 should be applicable */
    assert(winograd_applicable(3, 3, 1, 1, 1, 1) == true);
    printf(" PASS\n");
}

static void test_winograd_applicable_false(void) {
    printf("  test_winograd_applicable_false...");
    /* 5x5 kernel should not be applicable */
    assert(winograd_applicable(5, 5, 1, 1, 1, 1) == false);
    /* stride 2 should not be applicable */
    assert(winograd_applicable(3, 3, 2, 2, 1, 1) == false);
    /* dilation > 1 should not be applicable */
    assert(winograd_applicable(3, 3, 1, 1, 2, 2) == false);
    /* 1x1 kernel should not be applicable */
    assert(winograd_applicable(1, 1, 1, 1, 1, 1) == false);
    /* non-square kernel that is not 3x3 */
    assert(winograd_applicable(3, 5, 1, 1, 1, 1) == false);
    printf(" PASS\n");
}

static void test_select_variant(void) {
    printf("  test_select_variant...");

    /* Small spatial dimensions should select F2x2_3x3 */
    WinogradConfig cfg_small = winograd_select_variant(8, 8);
    assert(cfg_small.kernel_size == 3);
    assert(cfg_small.variant == WINOGRAD_F2x2_3x3 || cfg_small.variant == WINOGRAD_F4x4_3x3);
    assert(cfg_small.tile_size == 4 || cfg_small.tile_size == 6);
    assert(cfg_small.output_tile == 2 || cfg_small.output_tile == 4);

    /* Larger spatial dimensions may select F4x4_3x3 */
    WinogradConfig cfg_large = winograd_select_variant(56, 56);
    assert(cfg_large.kernel_size == 3);
    /* Either variant is valid; just ensure the config is consistent */
    if (cfg_large.variant == WINOGRAD_F2x2_3x3) {
        assert(cfg_large.tile_size == 4);
        assert(cfg_large.output_tile == 2);
    } else {
        assert(cfg_large.variant == WINOGRAD_F4x4_3x3);
        assert(cfg_large.tile_size == 6);
        assert(cfg_large.output_tile == 4);
    }
    printf(" PASS\n");
}

static void test_winograd_conv2d_basic(void) {
    printf("  test_winograd_conv2d_basic...");

    int batch = 1, in_c = 1, out_c = 1;
    int H = 8, W = 8;
    int pad_h = 1, pad_w = 1;
    int groups = 1;

    /* Allocate input: 1x1x8x8, all 1.0 */
    float* input = (float*)calloc(batch * in_c * H * W, sizeof(float));
    assert(input != NULL);
    for (int i = 0; i < batch * in_c * H * W; i++) {
        input[i] = 1.0f;
    }

    /* Allocate weight: 1x1x3x3, all 1.0 */
    float* weight = (float*)calloc(out_c * in_c * 3 * 3, sizeof(float));
    assert(weight != NULL);
    for (int i = 0; i < out_c * in_c * 3 * 3; i++) {
        weight[i] = 1.0f;
    }

    /* Output dimensions with padding=1: same as input -> 8x8 */
    int out_h = H;
    int out_w = W;
    float* output = (float*)calloc(batch * out_c * out_h * out_w, sizeof(float));
    assert(output != NULL);

    WinogradConfig config;
    config.variant = WINOGRAD_F2x2_3x3;
    config.tile_size = 4;
    config.output_tile = 2;
    config.kernel_size = 3;

    int ret = winograd_conv2d(input, weight, NULL, output,
                               batch, in_c, out_c, H, W,
                               pad_h, pad_w, groups, &config);
    assert(ret == 0);

    /*
     * For an all-1s 8x8 input with an all-1s 3x3 kernel and padding=1:
     * - Center pixels (not touching border) should sum to 9.0
     *   (all 9 kernel elements see a valid input of 1.0)
     * - Corner pixels should sum to 4.0
     * - Edge (non-corner) pixels should sum to 6.0
     *
     * Check a center pixel, e.g., output[3][3] at index 3*8+3 = 27
     */
    float center_val = output[3 * out_w + 3];
    assert(fabsf(center_val - 9.0f) < 1e-3f);

    /* Check a corner pixel: output[0][0] */
    float corner_val = output[0];
    assert(fabsf(corner_val - 4.0f) < 1e-3f);

    /* Check an edge pixel: output[0][1] */
    float edge_val = output[1];
    assert(fabsf(edge_val - 6.0f) < 1e-3f);

    free(input);
    free(weight);
    free(output);
    printf(" PASS\n");
}

int main(void) {
    printf("=== Winograd Convolution Tests ===\n");

    test_winograd_applicable_true();
    test_winograd_applicable_false();
    test_select_variant();
    test_winograd_conv2d_basic();

    printf("All Winograd tests passed.\n");
    return 0;
}
