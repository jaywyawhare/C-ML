#include "torch_compat.h"
#include "test_require.h"
#include <stdio.h>
#include <stdlib.h>

static float* realize(Tensor* t) {
    float* p = torch_tensor_data_ptr_f32(t);
    REQUIRE(p != NULL);
    return p;
}

static void test_creation_aliases(void) {
    printf("  test_creation_aliases...");

    TorchTensorOptions opts = torch_options();
    opts                    = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts                    = torch_options_device(opts, DEVICE_CPU);

    int shape[] = {2, 3};
    Tensor* z   = torch_zeros(shape, 2, &opts);
    REQUIRE(z != NULL);
    float* zd = realize(z);
    for (int i = 0; i < 6; i++)
        REQUIRE(zd[i] == 0.0f);
    torch_tensor_free(z);

    Tensor* o = torch_ones(shape, 2, &opts);
    REQUIRE(o != NULL);
    float* od = realize(o);
    for (int i = 0; i < 6; i++)
        REQUIRE(od[i] == 1.0f);
    torch_tensor_free(o);

    Tensor* r = torch_randn(shape, 2, &opts);
    REQUIRE(r != NULL);
    torch_tensor_free(r);

    Tensor* e = torch_eye(3, &opts);
    REQUIRE(e != NULL);
    torch_tensor_free(e);

    printf(" PASSED\n");
}

static void test_op_aliases(void) {
    printf("  test_op_aliases...");

    TorchTensorOptions opts = torch_options();
    opts                    = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts                    = torch_options_device(opts, DEVICE_CPU);

    int shape[] = {2, 2};
    Tensor* a   = torch_ones(shape, 2, &opts);
    Tensor* b   = torch_ones(shape, 2, &opts);
    REQUIRE(a && b);

    Tensor* c = torch_add(a, b);
    REQUIRE(c != NULL);
    float* cd = realize(c);
    for (int i = 0; i < 4; i++)
        REQUIRE(cd[i] == 2.0f);

    Tensor* d = torch_mul(a, b);
    REQUIRE(d != NULL);

    Tensor* m = torch_matmul(a, b);
    REQUIRE(m != NULL);

    torch_tensor_free(a);
    torch_tensor_free(b);
    torch_tensor_free(c);
    torch_tensor_free(d);
    torch_tensor_free(m);

    printf(" PASSED\n");
}

static void test_activation_aliases(void) {
    printf("  test_activation_aliases...");

    TorchTensorOptions opts = torch_options();
    opts                    = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts                    = torch_options_device(opts, DEVICE_CPU);

    int shape[] = {4};
    Tensor* x   = torch_ones(shape, 1, &opts);
    REQUIRE(x != NULL);
    float* xd = realize(x);
    xd[0]     = -1.0f;
    xd[1]     = 0.0f;
    xd[2]     = 1.0f;
    xd[3]     = 2.0f;

    Tensor* r = torch_relu(x);
    REQUIRE(r != NULL);
    float* rd = realize(r);
    REQUIRE(rd[0] == 0.0f);
    REQUIRE(rd[2] == 1.0f);
    torch_tensor_free(r);

    Tensor* s = torch_sigmoid(x);
    REQUIRE(s != NULL);
    torch_tensor_free(s);

    Tensor* t = torch_tanh(x);
    REQUIRE(t != NULL);
    torch_tensor_free(t);

    torch_tensor_free(x);

    printf(" PASSED\n");
}

int main(void) {
    torch_init();
    printf("Running torch compat tests:\n");

    test_creation_aliases();
    test_op_aliases();
    test_activation_aliases();

    torch_cleanup();
    printf("All torch compat tests passed.\n");
    return 0;
}
