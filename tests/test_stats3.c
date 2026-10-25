/* More descriptive statistics (ptp, iqr, mode, entropy, gmean, hmean), checked against
 * NumPy (np.ptp) and SciPy (scipy.stats.iqr/mode/entropy/gmean/hmean). */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static int test_ptp_iqr(void) {
    float d[] = {1, 2, 2, 3, 3, 3, 4};
    Tensor* D = cml_tensor_1d(d, 7);
    int ok    = fabsf(cml_ptp(D) - 3.0f) < 1e-6f && /* np.ptp */
             fabsf(cml_iqr(D) - 1.0f) < 1e-6f;      /* scipy.stats.iqr */
    tensor_free(D);
    return ok;
}

static int test_mode(void) {
    float d[] = {1, 2, 2, 3, 3, 3, 4};
    Tensor* D = cml_tensor_1d(d, 7);
    int ok    = fabsf(cml_mode(D) - 3.0f) < 1e-6f;
    tensor_free(D);
    return ok;
}

static int test_entropy(void) {
    float p[] = {1, 1, 1, 1};
    Tensor* P = cml_tensor_1d(p, 4);
    int ok    = fabsf(cml_entropy(P, 0.0f) - 1.38629436f) < 1e-5f && /* natural log */
             fabsf(cml_entropy(P, 2.0f) - 2.0f) < 1e-5f;             /* base 2 */
    tensor_free(P);
    return ok;
}

static int test_gmean_hmean(void) {
    float g[] = {1, 2, 4, 8};
    float h[] = {1, 2, 4};
    Tensor* G = cml_tensor_1d(g, 4);
    Tensor* H = cml_tensor_1d(h, 3);
    int ok    = fabsf(cml_gmean(G) - 2.82842712f) < 1e-5f && /* scipy.stats.gmean */
             fabsf(cml_hmean(H) - 1.71428571f) < 1e-5f;      /* scipy.stats.hmean */
    tensor_free(G);
    tensor_free(H);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== more descriptive statistics ===\n");
    TEST(ptp_iqr);
    TEST(mode);
    TEST(entropy);
    TEST(gmean_hmean);
    cml_cleanup();
    return TEST_SUMMARY();
}
