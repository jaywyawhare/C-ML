/* NumPy .npy I/O, validated against a real file produced by numpy.save (2x3 float32
 * [[0,1,2],[3,4,5]]) and a save->load round-trip. */
#include <math.h>
#include <stdio.h>
#include "cml.h"
#include "core/npy_io.h"
#include "tensor/tensor.h"
#include "test_harness.h"

/* Exact bytes written by numpy.save for np.arange(6,dtype='<f4').reshape(2,3). */
static const unsigned char NPY_REF[] = {
    147, 78,  85,  77,  80, 89, 1,  0,  118, 0,   123, 39,  100, 101, 115, 99,  114, 39,  58,
    32,  39,  60,  102, 52, 39, 44, 32, 39,  102, 111, 114, 116, 114, 97,  110, 95,  111, 114,
    100, 101, 114, 39,  58, 32, 70, 97, 108, 115, 101, 44,  32,  39,  115, 104, 97,  112, 101,
    39,  58,  32,  40,  50, 44, 32, 51, 41,  44,  32,  125, 32,  32,  32,  32,  32,  32,  32,
    32,  32,  32,  32,  32, 32, 32, 32, 32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,
    32,  32,  32,  32,  32, 32, 32, 32, 32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,
    32,  32,  32,  32,  32, 32, 32, 32, 32,  32,  32,  32,  32,  10,  0,   0,   0,   0,   0,
    0,   128, 63,  0,   0,  0,  64, 0,  0,   64,  64,  0,   0,   128, 64,  0,   0,   160, 64};

static int test_load_numpy_ref(void) {
    FILE* f = fopen("/tmp/cml_test_ref.npy", "wb");
    if (!f)
        return 0;
    fwrite(NPY_REF, 1, sizeof NPY_REF, f);
    fclose(f);
    Tensor* t = cml_tensor_load_npy("/tmp/cml_test_ref.npy");
    int ok = t && t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3 && t->dtype == DTYPE_FLOAT32;
    float* d = ok ? (float*)tensor_data_ptr(t) : NULL;
    for (int i = 0; ok && d && i < 6; i++)
        ok &= fabsf(d[i] - (float)i) < 1e-6f;
    tensor_free(t);
    return ok;
}

static int test_round_trip(void) {
    float data[] = {1.5f, -2.0f, 3.25f, 0.0f};
    Tensor* t    = cml_tensor_2d(data, 2, 2);
    int ok       = cml_tensor_save_npy(t, "/tmp/cml_test_rt.npy") == 0;
    Tensor* back = cml_tensor_load_npy("/tmp/cml_test_rt.npy");
    ok &= back && back->ndim == 2 && back->shape[0] == 2 && back->shape[1] == 2;
    float* d = ok ? (float*)tensor_data_ptr(back) : NULL;
    for (int i = 0; ok && d && i < 4; i++)
        ok &= fabsf(d[i] - data[i]) < 1e-6f;
    tensor_free(t);
    tensor_free(back);
    return ok;
}

static int test_rejects_bad(void) {
    return cml_tensor_load_npy("/tmp/does_not_exist_xyz.npy") == NULL;
}

int main(void) {
    cml_init();
    printf("=== NumPy .npy I/O ===\n");
    TEST(load_numpy_ref);
    TEST(round_trip);
    TEST(rejects_bad);
    cml_cleanup();
    return TEST_SUMMARY();
}
