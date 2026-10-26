/* Clustering round 2 (silhouette score, k-NN prediction), checked against sklearn. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_silhouette(void) {
    float x[]   = {0, 0, 0.1f, 0, 5, 5, 5.1f, 5};
    float lab[] = {0, 0, 1, 1};
    Tensor* X   = cml_tensor_2d(x, 4, 2);
    Tensor* L   = cml_tensor_1d(lab, 4);
    int ok      = fabsf(cml_silhouette_score(X, L) - 0.98585786f) < 1e-4f;
    tensor_free(X);
    tensor_free(L);
    return ok;
}

static int test_knn(void) {
    float xtr[] = {0, 0, 0, 1, 5, 5, 5, 6};
    float ytr[] = {0, 0, 1, 1};
    float xq[]  = {0.1f, 0.1f, 4.9f, 5.5f};
    Tensor* XTR = cml_tensor_2d(xtr, 4, 2);
    Tensor* YTR = cml_tensor_1d(ytr, 4);
    Tensor* XQ  = cml_tensor_2d(xq, 2, 2);
    Tensor* P1  = cml_knn_predict(XTR, YTR, XQ, 1);
    float q1[]  = {0.1f, 0.1f};
    Tensor* Q1  = cml_tensor_2d(q1, 1, 2);
    Tensor* P3  = cml_knn_predict(XTR, YTR, Q1, 3);
    int ok      = P1 && fabsf(at(P1, 0) - 0.0f) < 1e-6f && fabsf(at(P1, 1) - 1.0f) < 1e-6f && P3 &&
             fabsf(at(P3, 0) - 0.0f) < 1e-6f;
    tensor_free(XTR);
    tensor_free(YTR);
    tensor_free(XQ);
    tensor_free(Q1);
    tensor_free(P1);
    tensor_free(P3);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== clustering round 2 ===\n");
    TEST(silhouette);
    TEST(knn);
    cml_cleanup();
    return TEST_SUMMARY();
}
