/* Distance and similarity (cosine, euclidean, manhattan, pairwise cdist), checked against
 * scipy.spatial.distance. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "test_harness.h"

static float at(Tensor* t, int i) {
    float* d = (float*)tensor_data_ptr(t);
    return d ? d[i] : 0.0f;
}

static int test_vector_distances(void) {
    float a[] = {1, 2, 3}, b[] = {4, 5, 6};
    Tensor* A = cml_tensor_1d(a, 3);
    Tensor* B = cml_tensor_1d(b, 3);
    int ok    = fabsf(cml_cosine_similarity(A, B) - 0.97463185f) < 1e-5f &&
             fabsf(cml_euclidean_distance(A, B) - 5.19615242f) < 1e-5f &&
             fabsf(cml_manhattan_distance(A, B) - 9.0f) < 1e-5f;
    tensor_free(A);
    tensor_free(B);
    return ok;
}

static int test_cdist(void) {
    /* scipy.cdist -> [[0,3,4],[sqrt2, sqrt5, sqrt10]]. */
    float a[] = {0, 0, 1, 1};
    float b[] = {0, 0, 0, 3, 4, 0};
    Tensor* A = cml_tensor_2d(a, 2, 2);
    Tensor* B = cml_tensor_2d(b, 3, 2);
    Tensor* D = cml_cdist(A, B);
    float e[] = {0.0f, 3.0f, 4.0f, sqrtf(2.0f), sqrtf(5.0f), sqrtf(10.0f)};
    int ok    = D && D->shape[0] == 2 && D->shape[1] == 3;
    for (int i = 0; ok && i < 6; i++)
        ok &= fabsf(at(D, i) - e[i]) < 1e-5f;
    tensor_free(A);
    tensor_free(B);
    tensor_free(D);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== distance & similarity ===\n");
    TEST(vector_distances);
    TEST(cdist);
    cml_cleanup();
    return TEST_SUMMARY();
}
