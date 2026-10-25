/* HEVC reconstruction (H.265 §8.6.5): recSamples = Clip1(predSamples + resSamples).
 * Exactly verifiable: zero residual returns the prediction, a known residual adds,
 * and out-of-range sums saturate to [0,255]. */
#include <stdint.h>
#include <stdio.h>
#include "core/hevc.h"
#include "test_harness.h"

static int test_zero_residual(void) {
    uint8_t pred[16], out[16];
    int32_t res[16] = {0};
    for (int i = 0; i < 16; i++)
        pred[i] = (uint8_t)(i * 7 + 3);
    if (cml_hevc_reconstruct(pred, res, 4, out) != 0)
        return 0;
    for (int i = 0; i < 16; i++)
        if (out[i] != pred[i])
            return 0;
    return 1;
}

static int test_adds_residual(void) {
    uint8_t pred[16], out[16];
    int32_t res[16];
    for (int i = 0; i < 16; i++) {
        pred[i] = 100;
        res[i]  = i - 8; /* -8..7 */
    }
    if (cml_hevc_reconstruct(pred, res, 4, out) != 0)
        return 0;
    for (int i = 0; i < 16; i++)
        if (out[i] != (uint8_t)(100 + (i - 8)))
            return 0;
    return 1;
}

static int test_saturates(void) {
    uint8_t pred[4] = {10, 250, 0, 200}, out[4];
    int32_t res[4]  = {-50, 50, -1, 100}; /* -> -40, 300, -1, 300 */
    if (cml_hevc_reconstruct(pred, res, 2, out) != 0)
        return 0;
    return out[0] == 0 && out[1] == 255 && out[2] == 0 && out[3] == 255;
}

static int test_rejects_bad_args(void) {
    uint8_t pred[4] = {0}, out[4];
    int32_t res[4]  = {0};
    return cml_hevc_reconstruct(NULL, res, 2, out) == -1 &&
           cml_hevc_reconstruct(pred, res, 0, out) == -1;
}

int main(void) {
    printf("=== HEVC reconstruction ===\n");
    TEST(zero_residual);
    TEST(adds_residual);
    TEST(saturates);
    TEST(rejects_bad_args);
    return TEST_SUMMARY();
}
