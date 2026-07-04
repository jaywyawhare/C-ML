#include "torch/torch_c.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float* realize(Tensor* t) {
    float* p = torch_tensor_data_ptr_f32(t);
    assert(p != NULL);
    return p;
}

static void test_lifecycle(void) {
    printf("  test_lifecycle...");
    assert(torch_init() == 0);
    int major = 0, minor = 0, patch = 0;
    const char* ver = NULL;
    torch_get_version(&major, &minor, &patch, &ver);
    assert(ver != NULL);
    printf(" PASSED\n");
}

static void test_tensor_options(void) {
    printf("  test_tensor_options...");

    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts = torch_options_device(opts, DEVICE_CPU);
    opts = torch_options_requires_grad(opts, true);

    int shape[] = {2, 2};
    Tensor* t = torch_zeros(shape, 2, &opts);
    assert(t != NULL);
    assert(torch_tensor_ndim(t) == 2);
    assert(torch_tensor_numel(t) == 4);
    assert(torch_tensor_dtype(t) == DTYPE_FLOAT32);
    assert(torch_tensor_device(t) == DEVICE_CPU);
    assert(torch_tensor_requires_grad(t));
    assert(torch_tensor_sizes(t)[0] == 2);
    assert(torch_tensor_sizes(t)[1] == 2);

    float* data = realize(t);
    for (int i = 0; i < 4; i++)
        assert(data[i] == 0.0f);

    torch_tensor_free(t);
    printf(" PASSED\n");
}

static void test_tensor_ops(void) {
    printf("  test_tensor_ops...");

    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts = torch_options_device(opts, DEVICE_CPU);

    int shape[] = {2, 2};
    Tensor* a = torch_ones(shape, 2, &opts);
    Tensor* b = torch_ones(shape, 2, &opts);

    Tensor* c = torch_add(a, b);
    assert(c != NULL);
    float* cd = realize(c);
    for (int i = 0; i < 4; i++)
        assert(cd[i] == 2.0f);

    Tensor* m = torch_matmul(a, b);
    assert(m != NULL);

    Tensor* r = torch_relu(a);
    assert(r != NULL);

    Tensor* s = torch_softmax(a, 0);
    assert(s != NULL);

    torch_tensor_free(a);
    torch_tensor_free(b);
    torch_tensor_free(c);
    torch_tensor_free(m);
    torch_tensor_free(r);
    torch_tensor_free(s);
    printf(" PASSED\n");
}

static void test_autograd(void) {
    printf("  test_autograd...");

    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts = torch_options_device(opts, DEVICE_CPU);
    opts = torch_options_requires_grad(opts, true);

    int shape[] = {2};
    Tensor* x = torch_ones(shape, 1, &opts);
    Tensor* y = torch_mul(x, x);
    Tensor* loss = torch_sum(y, -1, false);

    torch_backward(loss, NULL, false, false);

    Tensor* grad = torch_get_grad(x);
    assert(grad != NULL);

    torch_tensor_free(x);
    torch_tensor_free(y);
    torch_tensor_free(loss);
    torch_reset_ir();
    printf(" PASSED\n");
}

static void test_module_api(void) {
    printf("  test_module_api...");

    Sequential* model = torch_nn_sequential();
    torch_nn_sequential_add(model, (Module*)torch_nn_linear(4, 1, true));
    torch_nn_sequential_add(model, (Module*)torch_nn_relu());

    torch_module_train((Module*)model);
    assert(torch_module_is_training((Module*)model));

    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts = torch_options_device(opts, DEVICE_CPU);

    int shape[] = {3, 4};
    Tensor* input = torch_randn(shape, 2, &opts);
    Tensor* output = torch_module_forward((Module*)model, input);
    assert(output != NULL);
    assert(torch_tensor_numel(output) == 3);

    StateDict* sd = torch_module_state_dict((Module*)model, "");
    assert(sd != NULL);
    assert(sd->count > 0);
    torch_state_dict_free(sd);

    TorchRuntimeModule* rt = torch_runtime_from_module((Module*)model);
    assert(rt != NULL);
    Tensor* rt_out = torch_runtime_forward(rt, input);
    assert(rt_out != NULL);
    torch_tensor_free(rt_out);
    torch_runtime_free(rt);

    torch_module_eval((Module*)model);
    assert(!torch_module_is_training((Module*)model));

    torch_tensor_free(input);
    torch_tensor_free(output);
    torch_reset_ir();
    module_free((Module*)model);
    printf(" PASSED\n");
}

static void test_retain(void) {
    printf("  test_retain...");

    TorchTensorOptions opts = torch_options();
    int shape[] = {1};
    Tensor* t = torch_ones(shape, 1, &opts);
    assert(torch_tensor_ref_count(t) == 1);
    torch_tensor_retain(t);
    assert(torch_tensor_ref_count(t) == 2);
    torch_tensor_free(t);
    assert(torch_tensor_ref_count(t) == 1);
    torch_tensor_free(t);
    printf(" PASSED\n");
}

int main(void) {
    printf("Running torch_c API tests:\n");

    test_lifecycle();
    test_tensor_options();
    test_tensor_ops();
    test_autograd();
    test_module_api();
    test_retain();

    torch_cleanup();
    printf("All torch_c tests passed.\n");
    return 0;
}
