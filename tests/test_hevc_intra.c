/* HEVC intra prediction (H.265 §8.4.4.2) verified against hand-computed spec
 * results: the DC average with its luma boundary smoothing, the planar bilinear,
 * the pure vertical/horizontal copies, a positive-angle diagonal, and a
 * negative-angle mode that exercises the inverse-angle reference extension. */
#include <stdint.h>
#include <stdio.h>
#include "core/hevc.h"
#include "test_harness.h"

/**
 * Ramp references: top[i] = 10+i, left[i] = 50+i (2n = 8 samples each), corner 99.
 * @param top (uint8_t*)
 * @param left (uint8_t*)
 * @param corner (uint8_t*)
 * @return (void)
 */
static void refs(uint8_t* top, uint8_t* left, uint8_t* corner) {
    for (int i = 0; i < 8; i++) {
        top[i]  = (uint8_t)(10 + i);
        left[i] = (uint8_t)(50 + i);
    }
    *corner = 99;
}

static int test_dc(void) {
    uint8_t top[8], left[8], corner, out[16];
    refs(top, left, &corner);
    if (cml_hevc_intra_predict(1, 4, top, left, corner, out) != 0)
        return 0;
    /* dc = (4 + sum(top[0..3]) + sum(left[0..3])) >> 3 = (4+46+206)>>3 = 32 */
    int ok = out[5] == 32 && out[10] == 32;              /* interior unfiltered */
    ok &= out[0] == 31;                                  /* (50 + 64 + 10 + 2) >> 2 */
    ok &= out[1] == 27 && out[2] == 27 && out[3] == 27;  /* (top[x] + 98) >> 2 */
    ok &= out[4] == 37 && out[8] == 37 && out[12] == 37; /* (left[y] + 98) >> 2 */
    return ok;
}

static int test_dc_constant(void) {
    uint8_t top[8], left[8], out[16];
    for (int i = 0; i < 8; i++)
        top[i] = left[i] = 100;
    if (cml_hevc_intra_predict(1, 4, top, left, 100, out) != 0)
        return 0;
    for (int i = 0; i < 16; i++)
        if (out[i] != 100)
            return 0;
    return 1;
}

static int test_planar_constant(void) {
    uint8_t top[8], left[8], out[16];
    for (int i = 0; i < 8; i++)
        top[i] = left[i] = 100;
    if (cml_hevc_intra_predict(0, 4, top, left, 100, out) != 0)
        return 0;
    for (int i = 0; i < 16; i++)
        if (out[i] != 100)
            return 0;
    return 1;
}

static int test_vertical(void) { /* mode 26 copies the row above */
    uint8_t top[8], left[8], corner, out[16];
    refs(top, left, &corner);
    if (cml_hevc_intra_predict(26, 4, top, left, corner, out) != 0)
        return 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (out[y * 4 + x] != top[x])
                return 0;
    return 1;
}

static int test_horizontal(void) { /* mode 10 copies the column to the left */
    uint8_t top[8], left[8], corner, out[16];
    refs(top, left, &corner);
    if (cml_hevc_intra_predict(10, 4, top, left, corner, out) != 0)
        return 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (out[y * 4 + x] != left[y])
                return 0;
    return 1;
}

static int test_diagonal_34(void) { /* angle 32: pred[x][y] = top[x+y+1] */
    uint8_t top[8], left[8], corner, out[16];
    refs(top, left, &corner);
    if (cml_hevc_intra_predict(34, 4, top, left, corner, out) != 0)
        return 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (out[y * 4 + x] != top[x + y + 1])
                return 0;
    return 1;
}

static int test_diagonal_2(void) { /* angle 32, horizontal class: out[i][j] = left[i+j+1] */
    uint8_t top[8], left[8], corner, out[16];
    refs(top, left, &corner);
    if (cml_hevc_intra_predict(2, 4, top, left, corner, out) != 0)
        return 0;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (out[i * 4 + j] != left[i + j + 1])
                return 0;
    return 1;
}

static int test_mode18_inv_angle(void) { /* angle -32: diagonal through the corner */
    uint8_t top[8], left[8], corner, out[16];
    refs(top, left, &corner);
    if (cml_hevc_intra_predict(18, 4, top, left, corner, out) != 0)
        return 0;
    int ok = out[0] == corner;       /* x==y hits p[-1][-1] */
    ok &= out[0 * 4 + 3] == top[2];  /* x>y: top[x-y-1] */
    ok &= out[3 * 4 + 0] == left[2]; /* x<y: left[y-x-1] via inverse angle */
    return ok;
}

static int test_rejects_bad_args(void) {
    uint8_t top[8] = {0}, left[8] = {0}, out[16];
    return cml_hevc_intra_predict(35, 4, top, left, 0, out) == -1 &&
           cml_hevc_intra_predict(0, 5, top, left, 0, out) == -1 &&
           cml_hevc_intra_predict(0, 4, NULL, left, 0, out) == -1;
}

int main(void) {
    printf("=== HEVC intra prediction ===\n");
    TEST(dc);
    TEST(dc_constant);
    TEST(planar_constant);
    TEST(vertical);
    TEST(horizontal);
    TEST(diagonal_34);
    TEST(diagonal_2);
    TEST(mode18_inv_angle);
    TEST(rejects_bad_args);
    return TEST_SUMMARY();
}
