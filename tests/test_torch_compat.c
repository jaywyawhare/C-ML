#include "torch_compat.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static float* realize(Tensor* t) {
    return (float*)tensor_data_ptr(t);
}

static void test_creation_aliases(void) {
    printf("  test_creation_aliases...");

    int shape[] = {2, 3};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    Tensor* z = torch_zeros(shape, 2, &cfg);
    assert(z != NULL);
    float* zd = realize(z);
    for (int i = 0; i < 6; i++) assert(zd[i] == 0.0f);
    tensor_free(z);

    Tensor* o = torch_ones(shape, 2, &cfg);
    assert(o != NULL);
    float* od = realize(o);
    for (int i = 0; i < 6; i++) assert(od[i] == 1.0f);
    tensor_free(o);

    Tensor* r = torch_randn(shape, 2, &cfg);
    assert(r != NULL);
    tensor_free(r);

    Tensor* e = torch_eye(3, &cfg);
    assert(e != NULL);
    tensor_free(e);

    printf(" PASSED\n");
}

static void test_op_aliases(void) {
    printf("  test_op_aliases...");

    int shape[] = {2, 2};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    Tensor* a = torch_ones(shape, 2, &cfg);
    Tensor* b = torch_ones(shape, 2, &cfg);
    assert(a && b);

    Tensor* c = torch_add(a, b);
    assert(c != NULL);
    float* cd = realize(c);
    for (int i = 0; i < 4; i++) assert(cd[i] == 2.0f);

    Tensor* d = torch_mul(a, b);
    assert(d != NULL);

    Tensor* m = torch_matmul(a, b);
    assert(m != NULL);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
    tensor_free(m);

    printf(" PASSED\n");
}

static void test_activation_aliases(void) {
    printf("  test_activation_aliases...");

    int shape[] = {4};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* x = torch_ones(shape, 1, &cfg);
    assert(x != NULL);
    float* xd = realize(x);
    xd[0] = -1.0f;
    xd[1] = 0.0f;
    xd[2] = 1.0f;
    xd[3] = 2.0f;

    Tensor* r = torch_relu(x);
    assert(r != NULL);
    float* rd = realize(r);
    assert(rd[0] == 0.0f);
    assert(rd[2] == 1.0f);
    tensor_free(r);

    Tensor* s = torch_sigmoid(x);
    assert(s != NULL);
    tensor_free(s);

    Tensor* t = torch_tanh(x);
    assert(t != NULL);
    tensor_free(t);

    tensor_free(x);

    printf(" PASSED\n");
}

int main(void) {
    printf("Running torch compat tests:\n");

    test_creation_aliases();
    test_op_aliases();
    test_activation_aliases();

    printf("All torch compat tests passed.\n");
    return 0;
}
