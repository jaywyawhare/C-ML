#include "torch/torch_c.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_ref_count_api(void) {
    printf("  test_ref_count_api...");
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

static void test_data_ptr_f32(void) {
    printf("  test_data_ptr_f32...");
    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    int shape[] = {2};
    Tensor* t = torch_ones(shape, 2, &opts);
    float* data = torch_tensor_data_ptr_f32(t);
    assert(data != NULL);
    assert(data[0] == 1.0f && data[1] == 1.0f);
    torch_tensor_free(t);
    printf(" PASSED\n");
}

static void test_data_ptr_f32_rejects_wrong_dtype(void) {
    printf("  test_data_ptr_f32_rejects_wrong_dtype...");
    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_INT32);
    int shape[] = {2};
    Tensor* t = torch_zeros(shape, 2, &opts);
    assert(torch_tensor_data_ptr_f32(t) == NULL);
    assert(torch_has_error());
    torch_tensor_free(t);
    printf(" PASSED\n");
}

static void test_materialized_vs_lazy_ir(void) {
    printf("  test_materialized_vs_lazy_ir...");
    torch_set_eager_mode(true);
    torch_no_grad();

    float vals[] = {1, 2, 3, 4};
    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    int shape[] = {2, 2};
    Tensor* a = torch_empty(shape, 2, &opts);
    float* ad = torch_tensor_data_ptr_f32(a);
    memcpy(ad, vals, sizeof(vals));

    Tensor* b = torch_empty(shape, 2, &opts);
    float* bd = torch_tensor_data_ptr_f32(b);
    memcpy(bd, vals, sizeof(vals));

    Tensor* s = torch_add(a, b);
    assert(torch_tensor_is_materialized(s));
    assert(!torch_tensor_has_lazy_ir(s));

    torch_set_eager_mode(false);
    torch_enable_grad();
    opts = torch_options_requires_grad(opts, true);
    Tensor* x = torch_ones(shape, 2, &opts);
    Tensor* y = torch_mul(x, x);
    assert(torch_tensor_has_lazy_ir(y));
    assert(!torch_tensor_is_materialized(y));

    torch_tensor_free(a);
    torch_tensor_free(b);
    torch_tensor_free(s);
    torch_tensor_free(x);
    torch_tensor_free(y);
    torch_reset_ir();
    torch_set_eager_mode(false);
    printf(" PASSED\n");
}

static void test_runtime_accessors(void) {
    printf("  test_runtime_accessors...");
    Sequential* model = torch_nn_sequential();
    torch_nn_sequential_add(model, (Module*)torch_nn_linear(4, 2, true));

    TorchRuntimeModule* rt = torch_runtime_from_module((Module*)model);
    assert(rt != NULL);
    assert(torch_runtime_get_kind(rt) == TORCH_RUNTIME_EAGER);
    assert(!torch_runtime_has_memory(rt));
    assert(torch_runtime_pte_arena_size(rt) == 0);
    torch_runtime_free(rt);
    module_free((Module*)model);
    printf(" PASSED\n");
}

static void test_nested_inference_mode(void) {
    printf("  test_nested_inference_mode...");
    torch_enable_grad();
    torch_inference_mode(true);
    torch_inference_mode(true);
    assert(!torch_is_grad_enabled());
    torch_inference_mode(false);
    assert(!torch_is_grad_enabled());
    torch_inference_mode(false);
    assert(torch_is_grad_enabled());
    assert(!torch_is_eager_mode());
    printf(" PASSED\n");
}

static void test_inference_mode_unmatched_exit(void) {
    printf("  test_inference_mode_unmatched_exit...");
    torch_enable_grad();
    assert(torch_is_grad_enabled());
    torch_inference_mode(false);
    assert(torch_is_grad_enabled());
    torch_no_grad();
    assert(!torch_is_grad_enabled());
    torch_inference_mode(false);
    assert(!torch_is_grad_enabled());
    torch_enable_grad();
    printf(" PASSED\n");
}

static void test_torch_realize_null(void) {
    printf("  test_torch_realize_null...");
    assert(torch_realize(NULL) == -1);
    printf(" PASSED\n");
}

static void test_inference_mode_restores_grad(void) {
    printf("  test_inference_mode_restores_grad...");
    torch_enable_grad();
    assert(torch_is_grad_enabled());
    torch_inference_mode(true);
    assert(!torch_is_grad_enabled());
    assert(torch_is_eager_mode());
    torch_inference_mode(false);
    assert(torch_is_grad_enabled());
    assert(!torch_is_eager_mode());
    printf(" PASSED\n");
}

static void test_clear_error(void) {
    printf("  test_clear_error...");
    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_INT32);
    int shape[] = {2};
    Tensor* t = torch_zeros(shape, 2, &opts);
    assert(torch_tensor_data_ptr_f32(t) == NULL);
    assert(torch_has_error());
    torch_clear_error();
    assert(!torch_has_error());
    torch_tensor_free(t);
    printf(" PASSED\n");
}

static void test_from_blob_null_rejected(void) {
    printf("  test_from_blob_null_rejected...");
    TorchTensorOptions opts = torch_options();
    int shape[] = {2, 2};
    assert(torch_from_blob(NULL, shape, 2, &opts) == NULL);
    assert(torch_has_error());
    torch_clear_error();
    printf(" PASSED\n");
}

static void test_item_float_scalar(void) {
    printf("  test_item_float_scalar...");
    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    int shape[] = {1};
    Tensor* t = torch_full(shape, 1, &opts, 3.5f);
    assert(torch_tensor_item_float(t) == 3.5f);
    torch_tensor_set_item_float(t, -1.0f);
    assert(torch_tensor_item_float(t) == -1.0f);
    torch_tensor_free(t);

    int shape2[] = {2};
    Tensor* bad = torch_zeros(shape2, 1, &opts);
    assert(torch_tensor_item_float(bad) == 0.0f);
    assert(torch_has_error());
    torch_clear_error();
    torch_tensor_free(bad);
    printf(" PASSED\n");
}

int main(void) {
    torch_init();
    printf("Running torch API accessor regression tests:\n");
    test_ref_count_api();
    test_data_ptr_f32();
    test_data_ptr_f32_rejects_wrong_dtype();
    test_materialized_vs_lazy_ir();
    test_runtime_accessors();
    test_inference_mode_restores_grad();
    test_nested_inference_mode();
    test_inference_mode_unmatched_exit();
    test_torch_realize_null();
    test_clear_error();
    test_from_blob_null_rejected();
    test_item_float_scalar();
    torch_cleanup();
    printf("All torch API accessor tests passed.\n");
    return 0;
}
