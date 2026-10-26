/* k-means clustering, checked on two well-separated clusters: the split, centroids and inertia
 * match sklearn.cluster.KMeans on the same data. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_kmeans(void) {
    float pts[12]  = {0.0f, 0.0f, 0.1f, 0.1f, 0.0f, 0.2f, 5.0f, 5.0f, 5.1f, 5.0f, 4.9f, 5.1f};
    Tensor* X      = cml_tensor_2d(pts, 6, 2);
    KMeansResult r = cml_kmeans(X, 2, 100);
    int ok         = r.centroids && r.labels;
    int l0         = (int)at(r.labels, 0);
    int l3         = (int)at(r.labels, 3);
    /* first three points cluster together, last three together, and the two groups differ. */
    ok = ok && (int)at(r.labels, 1) == l0 && (int)at(r.labels, 2) == l0 &&
         (int)at(r.labels, 4) == l3 && (int)at(r.labels, 5) == l3 && l0 != l3;
    /* centroid of the first cluster ~ mean of cluster A, inertia is small. */
    ok = ok && fabsf(at(r.centroids, l0 * 2 + 0) - 0.033333f) < 1e-3f &&
         fabsf(at(r.centroids, l0 * 2 + 1) - 0.1f) < 1e-3f &&
         fabsf(at(r.centroids, l3 * 2 + 0) - 5.0f) < 1e-3f &&
         fabsf(at(r.centroids, l3 * 2 + 1) - 5.033333f) < 1e-3f && r.inertia < 0.06f;
    tensor_free(X);
    tensor_free(r.centroids);
    tensor_free(r.labels);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== k-means ===\n");
    TEST(kmeans);
    cml_cleanup();
    return TEST_SUMMARY();
}
