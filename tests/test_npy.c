/* NumPy .npy I/O, validated against a real file produced by numpy.save (2x3 float32
 * [[0,1,2],[3,4,5]]) and a save->load round-trip. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

/* Property: for random float32 tensors of random rank and shape, save->load is
 * bit-exact (shape, rank and every element). Seeded so failures reproduce. */
static unsigned prng_state = 0x9E3779B9u;
static unsigned prng_next(void) {
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 17;
    prng_state ^= prng_state << 5;
    return prng_state;
}

static int test_round_trip_property(void) {
    for (int iter = 0; iter < 200; iter++) {
        int ndim = 1 + (int)(prng_next() % 4);
        int shape[4];
        size_t numel = 1;
        for (int d = 0; d < ndim; d++) {
            shape[d] = 1 + (int)(prng_next() % 6);
            numel *= (size_t)shape[d];
        }
        float* data = (float*)malloc(numel * sizeof(float));
        if (!data)
            return 0;
        for (size_t i = 0; i < numel; i++)
            data[i] = ((float)(int)(prng_next() % 20001) - 10000.0f) / 100.0f;

        Tensor* t  = cml_tensor(data, shape, ndim, NULL);
        int ok     = t && cml_tensor_save_npy(t, "/tmp/cml_test_prop.npy") == 0;
        Tensor* bk = ok ? cml_tensor_load_npy("/tmp/cml_test_prop.npy") : NULL;
        ok &= bk && bk->ndim == ndim && bk->numel == numel;
        for (int d = 0; ok && d < ndim; d++)
            ok &= bk->shape[d] == shape[d];
        float* out = ok ? (float*)tensor_data_ptr(bk) : NULL;
        for (size_t i = 0; ok && out && i < numel; i++)
            ok &= out[i] == data[i];

        tensor_free(t);
        tensor_free(bk);
        free(data);
        if (!ok) {
            printf("  property failed at iter %d (ndim=%d)\n", iter, ndim);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    cml_init();
    printf("=== NumPy .npy I/O ===\n");
    TEST(load_numpy_ref);
    TEST(round_trip);
    TEST(round_trip_property);
    TEST(rejects_bad);
    cml_cleanup();
    return TEST_SUMMARY();
}
