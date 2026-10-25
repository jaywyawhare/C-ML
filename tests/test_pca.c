/* PCA via covariance eigendecomposition, checked against sklearn.decomposition.PCA: the
 * explained-variance ratio is exact, components and projections match up to sign. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static float X[20] = {2.5f, 2.4f, 0.5f, 0.7f, 2.2f, 2.9f, 1.9f, 2.2f, 3.1f, 3.0f,
                      2.3f, 2.7f, 2.0f, 1.6f, 1.0f, 1.1f, 1.5f, 1.6f, 1.1f, 0.9f};

static int test_pca(void) {
    Tensor* M   = cml_tensor_2d(X, 10, 2);
    PCAResult p = cml_pca(M, 2);
    int ok      = p.components && p.explained_variance_ratio && p.transformed;
    /* explained-variance ratio is sign-independent and must match sklearn exactly. */
    ok = ok && fabsf(at(p.explained_variance_ratio, 0) - 0.96318131f) < 1e-4f &&
         fabsf(at(p.explained_variance_ratio, 1) - 0.03681869f) < 1e-4f;
    /* components [2,2]; magnitudes match sklearn (sign is arbitrary). */
    ok = ok && fabsf(fabsf(at(p.components, 0)) - 0.67787340f) < 1e-3f &&
         fabsf(fabsf(at(p.components, 1)) - 0.73517866f) < 1e-3f &&
         fabsf(fabsf(at(p.components, 2)) - 0.73517866f) < 1e-3f &&
         fabsf(fabsf(at(p.components, 3)) - 0.67787340f) < 1e-3f;
    /* first projected sample magnitudes match sklearn transform. */
    ok = ok && fabsf(fabsf(at(p.transformed, 0)) - 0.82797f) < 1e-3f &&
         fabsf(fabsf(at(p.transformed, 1)) - 0.175115f) < 1e-3f;
    tensor_free(M);
    tensor_free(p.components);
    tensor_free(p.explained_variance_ratio);
    tensor_free(p.transformed);
    return ok;
}

static int test_pca_one_component(void) {
    Tensor* M   = cml_tensor_2d(X, 10, 2);
    PCAResult p = cml_pca(M, 1);
    int ok      = p.components && p.components->shape[0] == 1 && p.transformed->shape[1] == 1 &&
             fabsf(at(p.explained_variance_ratio, 0) - 0.96318131f) < 1e-4f;
    tensor_free(M);
    tensor_free(p.components);
    tensor_free(p.explained_variance_ratio);
    tensor_free(p.transformed);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== PCA ===\n");
    TEST(pca);
    TEST(pca_one_component);
    cml_cleanup();
    return TEST_SUMMARY();
}
