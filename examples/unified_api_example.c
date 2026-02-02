/**
 * @file unified_api_example.c
 * @brief Example demonstrating unified API: User Layer + Tensor Level + UOp Level
 *
 * This example shows how all three abstraction levels work together:
 * 1. User-defined layers (PyTorch-like)
 * 2. High-level tensor operations (tensor_*)
 * 3. Low-level uops (tinygrad-like)
 *
 * All support autograd and IR capture!
 */

#include "cml.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    Module base;
} SimpleLayer;

static Tensor* simple_layer_forward(Module* module, Tensor* input) {
    (void)module;

    int scalar_shape[]  = {1};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* scalar = tensor_empty(scalar_shape, 1, &config);
    tensor_set_float(scalar, 0, 2.0f);

    Tensor* doubled = tensor_mul(input, scalar);
    Tensor* result  = uop_exp(doubled);

    tensor_free(scalar);
    tensor_free(doubled);
    return result;
}

static void example_tensor_level(void) {
    CMLIR_t ir = cml_ir_new(IR_TARGET_CUDA);
    cml_ir_enable_auto_capture(ir);

    // Perform tensor operations to capture
    int shape[]         = {3};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* x = tensor_empty(shape, 1, &config);
    Tensor* y = tensor_empty(shape, 1, &config);

    float* x_data = (float*)tensor_data_ptr(x);
    float* y_data = (float*)tensor_data_ptr(y);
    for (int i = 0; i < 3; i++) {
        x_data[i] = (float)(i + 1) * 0.5f;
        y_data[i] = (float)(i + 2) * 0.3f;
    }

    Tensor* z = tensor_add(x, y);
    Tensor* w = tensor_mul(z, x);

    if (w)
        tensor_free(w);
    if (z)
        tensor_free(z);
    tensor_free(y);
    tensor_free(x);

    char* code = cml_ir_compile(ir, NULL);
    if (code) {
        printf("Generated CUDA code:\n%s\n", code);
        free(code);
    }

    cml_ir_disable_auto_capture();
    cml_ir_free(ir);
}

static void example_uop_level(void) {
    CMLIR_t ir = cml_ir_new(IR_TARGET_METAL);

    // Add operations directly to IR using uops
    int shape[]         = {2};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* a = tensor_empty(shape, 1, &config);
    Tensor* b = tensor_empty(shape, 1, &config);

    float* a_data = (float*)tensor_data_ptr(a);
    float* b_data = (float*)tensor_data_ptr(b);
    a_data[0]     = 1.0f;
    a_data[1]     = 2.0f;
    b_data[0]     = 3.0f;
    b_data[1]     = 4.0f;

    Tensor* inputs[] = {a, b};
    cml_ir_add_uop(ir, UOP_ADD, inputs, 2, NULL);
    cml_ir_add_uop(ir, UOP_MUL, inputs, 2, NULL);
    cml_ir_add_uop(ir, UOP_EXP, inputs, 1, NULL);

    tensor_free(b);
    tensor_free(a);

    char* code = cml_ir_compile(ir, NULL);
    if (code) {
        printf("Generated Metal code:\n%s\n", code);
        free(code);
    }

    cml_ir_free(ir);
}

static void example_mixed_levels(void) { (void)simple_layer_forward; }

int main(void) {
    cml_init();
    cml_set_log_level(LOG_LEVEL_INFO);

    example_tensor_level();
    example_uop_level();
    example_mixed_levels();

    cml_cleanup();
    return 0;
}
