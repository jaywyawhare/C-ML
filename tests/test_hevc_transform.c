/* HEVC core transforms (H.265 §8.6.4) verified intrinsically — no reference
 * stream or external table, which is what blocked the rest of the decoder.
 *
 * The spec DCT-II matrices have exactly orthogonal rows (M·Mᵀ is diagonal), the
 * discriminator that separates the real tables from a close-but-wrong guess. The
 * rows are orthogonal but not equal-norm (so the transform is lossy by design),
 * hence correctness is: (a) every off-diagonal of M·Mᵀ is exactly 0, and (b) the
 * forward transform M·B·Mᵀ reconstructs B through the norm-normalized inverse.
 * One wrong entry breaks (a). */
#include <math.h>
#include <stdlib.h>
#include "core/hevc.h"
#include "test_harness.h"

/* exact: off-diagonal of M·Mᵀ (row dot products) must be 0. */
static int check_row_orthogonal(int n) {
    const int16_t* m = cml_hevc_transform_matrix(n, 0);
    if (!m)
        return 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            long dot = 0;
            for (int k = 0; k < n; k++)
                dot += (long)m[i * n + k] * m[j * n + k];
            if (i == j) {
                if (dot <= 0)
                    return 0;
            } else if (dot != 0) {
                return 0;
            }
        }
    }
    return 1;
}

/* forward C = M B Mᵀ, then reconstruct in double via the norm-normalized inverse
 * B'[i][j] = Σ_{p,q} (M[p][i]/Dp) C[p][q] (M[q][j]/Dq), require B' ≈ B. */
static int check_reconstructs(int n) {
    const int16_t* m = cml_hevc_transform_matrix(n, 0);
    if (!m)
        return 0;
    int nn = n * n;
    int32_t b[64], c[64];
    unsigned seed = 0xBEEFu + (unsigned)n;
    for (int i = 0; i < nn; i++) {
        seed = seed * 1103515245u + 12345u;
        b[i] = (int)((seed >> 16) % 201) - 100;
    }
    if (cml_hevc_forward_transform(b, c, n, 0) != 0)
        return 0;
    double D[8];
    for (int p = 0; p < n; p++) {
        double s = 0;
        for (int k = 0; k < n; k++)
            s += (double)m[p * n + k] * m[p * n + k];
        D[p] = s;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double acc = 0;
            for (int p = 0; p < n; p++)
                for (int q = 0; q < n; q++)
                    acc += (m[p * n + i] / D[p]) * (double)c[p * n + q] * (m[q * n + j] / D[q]);
            if (fabs(acc - (double)b[i * n + j]) > 1e-6)
                return 0;
        }
    return 1;
}

static int test_dct4_row_orthogonal(void) { return check_row_orthogonal(4); }
static int test_dct4_reconstructs(void) { return check_reconstructs(4); }
static int test_unsupported_rejected(void) {
    return cml_hevc_transform_matrix(8, 0) == NULL && cml_hevc_transform_matrix(16, 0) == NULL &&
           cml_hevc_transform_matrix(4, 1) == NULL;
}

int main(void) {
    printf("=== HEVC core transforms ===\n");
    TEST(dct4_row_orthogonal);
    TEST(dct4_reconstructs);
    TEST(unsupported_rejected);
    return TEST_SUMMARY();
}
