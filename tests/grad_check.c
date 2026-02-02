#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "autograd/autograd.h"
#include "core/logging.h"

// Epsilon for finite differences
#define EPSILON 1e-3f
#define TOLERANCE 1e-2f

// Helper to create random tensor
Tensor* random_tensor(int* shape, int ndim, bool requires_grad) {
    TensorConfig config = tensor_config_default();
    Tensor* t = tensor_empty(shape, ndim, &config);
    if (t) {
        float* data = (float*)tensor_data_ptr(t);
        for (size_t i = 0; i < t->numel; i++) {
            data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // -1 to 1
        }
        t->requires_grad = requires_grad;
    }
    return t;
}

// Helper to check gradients
bool check_gradient(const char* op_name, Tensor* input, Tensor* output, Tensor* grad_out) {
    printf("Checking gradient for %s...\n", op_name);

    if (!input->grad) {
        printf("  FAIL: No gradient computed for input\n");
        return false;
    }

    // Suppress unused variable warnings for now
    (void)output;
    (void)grad_out;

    return true; // Placeholder
}

// Specific check for UOP_LOG
bool check_log() {
    int shape[] = {2, 3};
    Tensor* x = random_tensor(shape, 2, true);
    // Ensure positive inputs for log
    float* data = (float*)tensor_data_ptr(x);
    for (size_t i = 0; i < x->numel; i++) data[i] = fabsf(data[i]) + 0.1f;

    Tensor* y = uop_log(x);

    // Backward
    TensorConfig config = tensor_config_default();
    Tensor* grad_out = tensor_ones(y->shape, y->ndim, &config);
    y->grad = grad_out;

    // Build backward graph
    if (y->ir_node) {
        extern int cml_ir_build_backward(CMLIR_t, struct IRNode*);
        extern int cml_ir_execute_backward(CMLIR_t);

        cml_ir_build_backward(y->ir_context, y->ir_node);
        cml_ir_execute_backward(y->ir_context);

        float* grad = (float*)tensor_data_ptr(x->grad);
        float* inp = (float*)tensor_data_ptr(x);

        for (size_t i = 0; i < x->numel; i++) {
            float expected = 1.0f / inp[i]; // Since grad_out is ones
            if (fabsf(grad[i] - expected) > TOLERANCE) {
                printf("FAIL: Log gradient mismatch at %zu. Got %f, expected %f\n", i, grad[i], expected);
                return false;
            }
        }
        printf("PASS: Log gradient check\n");
        return true;
    } else {
        printf("FAIL: Log - Output IR node not created. y->ir_node=%p\n", (void*)y->ir_node);
    }
    return false;
}

// Specific check for UOP_EXPAND
bool check_expand() {
    int shape[] = {1, 3};
    Tensor* x = random_tensor(shape, 2, true);

    int new_shape[] = {2, 3};
    ExpandParams params = {.new_shape = new_shape, .new_ndim = 2};
    Tensor* y = uop_expand(x, &params);

    TensorConfig config = tensor_config_default();
    Tensor* grad_out = tensor_ones(y->shape, y->ndim, &config); // Ones
    y->grad = grad_out;

    if (y->ir_node) {
        extern int cml_ir_build_backward(CMLIR_t, struct IRNode*);
        extern int cml_ir_execute_backward(CMLIR_t);

        cml_ir_build_backward(y->ir_context, y->ir_node);
        cml_ir_execute_backward(y->ir_context);

        float* grad = (float*)tensor_data_ptr(x->grad);

        // Expected: sum of ones over the expanded dimension (dim 0, size 2)
        // So expected gradient is 2.0 for all elements
        for (size_t i = 0; i < x->numel; i++) {
            if (fabsf(grad[i] - 2.0f) > TOLERANCE) {
                printf("FAIL: Expand gradient mismatch at %zu. Got %f, expected 2.0\n", i, grad[i]);
                return false;
            }
        }
        printf("PASS: Expand gradient check\n");
        return true;
    } else {
        printf("FAIL: Expand - Output IR node not created. y->ir_node=%p\n", (void*)y->ir_node);
    }
    return false;
}

// Specific check for UOP_MEAN
bool check_mean() {
    int shape[] = {2, 3};
    Tensor* x = random_tensor(shape, 2, true);

    ReduceParams params = {.dims = NULL, .num_dims = 0, .keepdim = false}; // Global mean
    Tensor* y = uop_mean(x, &params);

    TensorConfig config = tensor_config_default();
    Tensor* grad_out = tensor_ones(y->shape, y->ndim, &config);
    y->grad = grad_out;

    if (y->ir_node) {
        extern int cml_ir_build_backward(CMLIR_t, struct IRNode*);
        extern int cml_ir_execute_backward(CMLIR_t);

        cml_ir_build_backward(y->ir_context, y->ir_node);
        cml_ir_execute_backward(y->ir_context);

        float* grad = (float*)tensor_data_ptr(x->grad);

        // Expected: 1/N where N=6
        float expected = 1.0f / 6.0f;
        for (size_t i = 0; i < x->numel; i++) {
            if (fabsf(grad[i] - expected) > TOLERANCE) {
                printf("FAIL: Mean gradient mismatch at %zu. Got %f, expected %f\n", i, grad[i], expected);
                return false;
            }
        }
        printf("PASS: Mean gradient check\n");
        return true;
    } else {
        printf("FAIL: Mean - Output IR node not created. y->ir_node=%p\n", (void*)y->ir_node);
    }
    return false;
}

// Specific check for UOP_CONV2D
bool check_conv2d() {
    // Input: 1x1x3x3
    int in_shape[] = {1, 1, 3, 3};
    Tensor* x = random_tensor(in_shape, 4, true);
    float* x_data = (float*)tensor_data_ptr(x);
    for(size_t i=0; i<x->numel; i++) x_data[i] = (float)i;

    // Weight: 1x1x2x2
    int w_shape[] = {1, 1, 2, 2};
    Tensor* w = random_tensor(w_shape, 4, true);
    float* w_data = (float*)tensor_data_ptr(w);
    for(size_t i=0; i<w->numel; i++) w_data[i] = 1.0f;

    Conv2DParams params = {0}; // Default stride 1, pad 0

    Tensor* y = uop_conv2d(x, w, NULL, &params);
    if (!y) {
        printf("FAIL: Conv2D - Forward pass failed\n");
        return false;
    }

    if (!y->ir_node) {
        printf("FAIL: Conv2D - Output IR node not created\n");
        return false;
    }

    printf("Conv2D output node created. IR Node: %p\n", (void*)y->ir_node);

    // Output shape should be 1x1x2x2

    TensorConfig config = tensor_config_default();
    Tensor* grad_out = tensor_ones(y->shape, y->ndim, &config);
    y->grad = grad_out;

    if (y->ir_node) {
        extern int cml_ir_build_backward(CMLIR_t, struct IRNode*);
        extern int cml_ir_execute_backward(CMLIR_t);

        cml_ir_build_backward(y->ir_context, y->ir_node);
        cml_ir_execute_backward(y->ir_context);

        // Check gradient w.r.t input
        // Since weights are all 1s, and grad_out is all 1s
        // grad_input at (i,j) is number of times (i,j) participates in convolution
        // For 3x3 input, 2x2 kernel, stride 1:
        // (0,0): 1 (top-left of first window)
        // (0,1): 2 (top-right of first, top-left of second)
        // (0,2): 1 (top-right of second)
        // (1,0): 2
        // (1,1): 4
        // (1,2): 2
        // (2,0): 1
        // (2,1): 2
        // (2,2): 1

        float expected_grad[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
        float* grad = (float*)tensor_data_ptr(x->grad);

        for (int i = 0; i < 9; i++) {
            if (fabsf(grad[i] - expected_grad[i]) > TOLERANCE) {
                printf("FAIL: Conv2D input gradient mismatch at %d. Got %f, expected %f\n", i, grad[i], expected_grad[i]);
                return false;
            }
        }
        printf("PASS: Conv2D gradient check\n");
        return true;
    } else {
        printf("FAIL: Conv2D - Output IR node not created. y->ir_node=%p\n", (void*)y->ir_node);
    }
    return false;
}

int main() {
    printf("Running gradient checks...\n");
    bool all_passed = true;

    if (!check_log()) all_passed = false;
    if (!check_expand()) all_passed = false;
    if (!check_mean()) all_passed = false;
    if (!check_conv2d()) all_passed = false;

    if (all_passed) {
        printf("All checks PASSED!\n");
        return 0;
    } else {
        printf("Some checks FAILED.\n");
        return 1;
    }
}
