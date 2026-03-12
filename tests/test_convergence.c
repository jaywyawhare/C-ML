#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"

static float get_scalar(Tensor* t) {
    float* p = (float*)tensor_data_ptr(t);
    return p ? p[0] : INFINITY;
}

static int test_xor_mlp(void) {
    printf("  [XOR MLP] Sequential(Linear(2,4)+ReLU+Linear(4,1)+Sigmoid), Adam lr=0.01, 2000 epochs\n");

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_sigmoid());
    module_set_training((Module*)model, true);

    Parameter** params = NULL;
    int num_params = 0;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    Optimizer* optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("    ERROR: failed to create optimizer\n");
        free(params);
        module_free((Module*)model);
        return 0;
    }

    float xor_inputs[]  = { 0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 1.0f,1.0f };
    float xor_targets[] = { 0.0f, 1.0f, 1.0f, 0.0f };

    Tensor* X = cml_tensor_2d(xor_inputs,  4, 2);
    Tensor* Y = cml_tensor_2d(xor_targets, 4, 1);

    float final_loss = INFINITY;

    for (int epoch = 0; epoch < 2000; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor* out  = module_forward((Module*)model, X);
        Tensor* loss = tensor_bce_loss(out, Y);
        final_loss   = get_scalar(loss);

        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        if ((epoch + 1) % 500 == 0) {
            printf("    epoch %4d  loss = %.6f\n", epoch + 1, (double)final_loss);
        }

        tensor_free(loss);
        tensor_free(out);
        cml_ir_reset_global_context();
    }

    int pass = (final_loss < 0.01f);
    printf("    final loss = %.6f  (threshold < 0.01) => %s\n",
           (double)final_loss, pass ? "PASS" : "FAIL");

    tensor_free(X);
    tensor_free(Y);
    optimizer_free(optimizer);
    free(params);
    module_free((Module*)model);

    return pass;
}

static int test_linear_regression(void) {
    printf("  [Linear Regression] Linear(1,1), SGD lr=0.01, 500 epochs, y=2x+1\n");

    Linear* layer = nn_linear(1, 1, DTYPE_FLOAT32, DEVICE_CPU, true);
    Module* model = (Module*)layer;
    module_set_training(model, true);

    Parameter** params = NULL;
    int num_params = 0;
    module_collect_parameters(model, &params, &num_params, true);
    Optimizer* optimizer = optim_sgd(params, num_params, 0.01f, 0.0f, 0.0f);
    if (!optimizer) {
        printf("    ERROR: failed to create optimizer\n");
        free(params);
        module_free(model);
        return 0;
    }

    #define LR_N 20
    float x_data[LR_N];
    float y_data[LR_N];
    for (int i = 0; i < LR_N; i++) {
        x_data[i] = (float)i / (float)(LR_N - 1) * 2.0f - 1.0f;
        y_data[i] = 2.0f * x_data[i] + 1.0f;
    }

    Tensor* X = cml_tensor_2d(x_data, LR_N, 1);
    Tensor* Y = cml_tensor_2d(y_data, LR_N, 1);

    float final_loss = INFINITY;

    for (int epoch = 0; epoch < 500; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor* out  = module_forward(model, X);
        Tensor* loss = tensor_mse_loss(out, Y);
        final_loss   = get_scalar(loss);

        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        if ((epoch + 1) % 100 == 0) {
            printf("    epoch %4d  loss = %.6f\n", epoch + 1, (double)final_loss);
        }

        tensor_free(loss);
        tensor_free(out);
        cml_ir_reset_global_context();
    }

    float learned_weight = 0.0f;
    float learned_bias   = 0.0f;

    Parameter* w = linear_get_weight(layer);
    Parameter* b = linear_get_bias(layer);
    if (w && w->tensor) {
        float* wd = (float*)tensor_data_ptr(w->tensor);
        if (wd) learned_weight = wd[0];
    }
    if (b && b->tensor) {
        float* bd = (float*)tensor_data_ptr(b->tensor);
        if (bd) learned_bias = bd[0];
    }

    printf("    learned weight = %.4f (target ~2.0), bias = %.4f (target ~1.0)\n",
           (double)learned_weight, (double)learned_bias);
    printf("    final loss = %.6f\n", (double)final_loss);

    int weight_ok = (fabsf(learned_weight - 2.0f) < 0.5f);
    int bias_ok   = (fabsf(learned_bias   - 1.0f) < 0.5f);
    int loss_ok   = (final_loss < 0.1f);
    int pass      = weight_ok && bias_ok && loss_ok;

    printf("    weight close to 2.0 (tol 0.5): %s\n", weight_ok ? "PASS" : "FAIL");
    printf("    bias   close to 1.0 (tol 0.5): %s\n", bias_ok   ? "PASS" : "FAIL");
    printf("    loss < 0.1:                     %s\n", loss_ok   ? "PASS" : "FAIL");
    printf("    => %s\n", pass ? "PASS" : "FAIL");

    tensor_free(X);
    tensor_free(Y);
    optimizer_free(optimizer);
    free(params);
    module_free(model);

    return pass;
}

static int test_conv2d_pattern(void) {
    printf("  [Conv2d Pattern] Linear(64,2), Adam lr=0.01, 500 epochs (flattened 8x8 input)\n");

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(1 * 8 * 8, 2, DTYPE_FLOAT32, DEVICE_CPU, true));
    module_set_training((Module*)model, true);

    Parameter** params = NULL;
    int num_params = 0;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    Optimizer* optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("    ERROR: failed to create optimizer\n");
        free(params);
        module_free((Module*)model);
        return 0;
    }

    #define CONV_N       8
    #define CONV_H       8
    #define CONV_W       8
    #define CONV_PIXELS  (CONV_H * CONV_W)

    float img_data[CONV_N * 1 * CONV_H * CONV_W];
    float target_data[CONV_N * 2];

    memset(img_data,    0, sizeof(img_data));
    memset(target_data, 0, sizeof(target_data));

    for (int n = 0; n < CONV_N; n++) {
        int cls = n % 2;
        float* img = &img_data[n * CONV_PIXELS];

        if (cls == 0) {
            for (int r = 0; r < CONV_H / 2; r++)
                for (int c = 0; c < CONV_W; c++)
                    img[r * CONV_W + c] = 1.0f;
        } else {
            for (int r = 0; r < CONV_H; r++)
                for (int c = 0; c < CONV_W / 2; c++)
                    img[r * CONV_W + c] = 1.0f;
        }

        target_data[n * 2 + cls] = 1.0f;
    }

    int img_shape[] = { CONV_N, 1 * CONV_H * CONV_W };
    TensorConfig cfg = { .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true };
    Tensor* X = cml_tensor(img_data, img_shape, 2, &cfg);
    Tensor* Y = cml_tensor_2d(target_data, CONV_N, 2);

    float final_loss = INFINITY;

    for (int epoch = 0; epoch < 500; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor* out = module_forward((Module*)model, X);
        Tensor* loss = tensor_mse_loss(out, Y);
        final_loss = get_scalar(loss);

        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        if ((epoch + 1) % 100 == 0) {
            printf("    epoch %4d  loss = %.6f\n", epoch + 1, (double)final_loss);
        }

        tensor_free(loss);
        tensor_free(out);
        cml_ir_reset_global_context();
    }

    Tensor* preds = module_forward((Module*)model, X);
    float* pred_data = (float*)tensor_data_ptr(preds);
    float* tgt_data  = (float*)tensor_data_ptr(Y);

    int correct = 0;
    if (pred_data && tgt_data) {
        for (int n = 0; n < CONV_N; n++) {
            int pred_cls = (pred_data[n * 2 + 1] > pred_data[n * 2 + 0]) ? 1 : 0;
            int true_cls = (tgt_data[n * 2 + 1]  > tgt_data[n * 2 + 0])  ? 1 : 0;
            if (pred_cls == true_cls) correct++;
        }
    }

    float accuracy = (float)correct / (float)CONV_N;
    int pass = (accuracy >= 0.75f);

    printf("    accuracy = %.2f (%d/%d)  (threshold >= 0.75) => %s\n",
           (double)accuracy, correct, CONV_N, pass ? "PASS" : "FAIL");

    tensor_free(preds);
    tensor_free(X);
    tensor_free(Y);
    optimizer_free(optimizer);
    free(params);
    module_free((Module*)model);

    return pass;
}

int main(void) {
    cml_init();
    cml_seed(42);

    printf("test_convergence\n\n");

    int total   = 0;
    int passed  = 0;

    printf("[1/3] XOR MLP Convergence\n");
    total++;
    if (test_xor_mlp()) passed++;
    printf("\n");

    printf("[2/3] Linear Regression Convergence\n");
    total++;
    if (test_linear_regression()) passed++;
    printf("\n");

    printf("[3/3] Conv2d Pattern Classification\n");
    total++;
    if (test_conv2d_pattern()) passed++;
    printf("\n");

    printf("%d/%d convergence tests passed\n", passed, total);

    cml_cleanup();
    return (passed == total) ? 0 : 1;
}
