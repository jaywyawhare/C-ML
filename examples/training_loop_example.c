/**
 * @file training_loop_example.c
 * @brief Minimal training-loop example using Sequential, Optimizer, and MSE loss.
 */

#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void training_example(void) {
    autograd_init();
    autograd_set_grad_mode(true);

    int input_size  = 10;
    int hidden_size = 20;
    int output_size = 1;
    Sequential* seq = nn_sequential();
    sequential_add(seq,
                   (Module*)nn_linear(input_size, hidden_size, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(seq, (Module*)nn_relu(false));
    sequential_add(seq,
                   (Module*)nn_linear(hidden_size, output_size, DTYPE_FLOAT32, DEVICE_CPU, true));
    Module* model = (Module*)seq;

    if (!model) {
        LOG_ERROR("Failed to create model");
        return;
    }

    module_set_training(model, true);

    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(model, &params, &num_params, true) != 0) {
        LOG_ERROR("Failed to collect model parameters");
        module_free(model);
        return;
    }

    printf("Model has %d parameters\n", num_params);

    Optimizer* optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

    if (!optimizer) {
        LOG_ERROR("Failed to create optimizer");
        CM_FREE(params);
        module_free(model);
        return;
    }

    printf("Created %s optimizer\n", optimizer_get_name(optimizer));

    int batch_size  = 32;
    int num_samples = 100;

    int input_shape[]  = {batch_size, input_size};
    Tensor* inputs     = tensor_empty(input_shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
    int target_shape[] = {batch_size, output_size};
    Tensor* targets    = tensor_zeros(target_shape, 1, DTYPE_FLOAT32, DEVICE_CPU);

    float* input_data  = (float*)tensor_data_ptr(inputs);
    float* target_data = (float*)tensor_data_ptr(targets);
    for (int i = 0; i < batch_size * input_size; i++) {
        input_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    for (int i = 0; i < batch_size; i++) {
        target_data[i] = ((float)rand() / RAND_MAX);
    }

    int num_epochs        = 100;
    float best_loss       = INFINITY;
    int patience          = 0;
    int no_improve_epochs = 5;
    float improvement_tol = 1e-5f;
    int lr_step_size      = 20;
    float lr_gamma        = 0.5f;

    printf("\nStarting training for %d epochs...\n\n", num_epochs);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < num_samples / batch_size; batch++) {
            optimizer_zero_grad(optimizer);

            Tensor* outputs = module_forward(model, inputs);

            if (!outputs) {
                LOG_ERROR("Forward pass failed");
                continue;
            }

            Tensor* loss = tensor_mse_loss(outputs, targets);

            if (!loss) {
                LOG_ERROR("Loss computation failed");
                continue;
            }

            if (batch == 0) {
                float p0 = tensor_get_float(outputs, 0);
                float p1 = tensor_get_float(outputs, 1);
                float p2 = tensor_get_float(outputs, 2);
                float t0 = tensor_get_float(targets, 0);
                float t1 = tensor_get_float(targets, 1);
                float t2 = tensor_get_float(targets, 2);
                printf("  sample preds: [%.3f, %.3f, %.3f]  targets: [%.3f, %.3f, %.3f]\n", p0, p1,
                       p2, t0, t1, t2);
            }

            tensor_backward(loss, NULL, false, false);

            float* loss_data = (float*)tensor_data_ptr(loss);
            if (loss_data) {
                epoch_loss += loss_data[0];
            }

            optimizer_step(optimizer);

            if (loss)
                tensor_free(loss);
            if (outputs)
                tensor_free(outputs);
        }

        epoch_loss /= (num_samples / batch_size);
        printf("Epoch %d/%d - Loss: %.6f\n", epoch + 1, num_epochs, epoch_loss);

        if (epoch_loss + improvement_tol < best_loss) {
            best_loss         = epoch_loss;
            no_improve_epochs = 0;
        } else {
            no_improve_epochs++;
            if (no_improve_epochs >= patience) {
                printf("Early stopping at epoch %d (best loss %.6f)\n", epoch + 1, best_loss);
                break;
            }
        }

        if ((epoch + 1) % lr_step_size == 0) {
            float current_lr = optimizer_get_group_lr(optimizer, 0);
            float new_lr     = current_lr * lr_gamma;
            optimizer_set_lr(optimizer, new_lr);
            printf("  LR decayed to %.6f\n", new_lr);
        }
    }

    printf("\nTraining completed!\n");

    Tensor* eval_out = module_forward(model, inputs);
    if (eval_out) {
        printf("\nEval snapshot (first 5):\n");
        size_t limit = eval_out->numel;
        if (limit > targets->numel)
            limit = targets->numel;
        if (limit > 5)
            limit = 5;
        for (size_t i = 0; i < limit; i++) {
            printf("  %zu: pred=%.4f  target=%.4f\n", i, tensor_get_float(eval_out, i),
                   tensor_get_float(targets, i));
        }
        tensor_free(eval_out);
    }

    tensor_free(targets);
    tensor_free(inputs);
    optimizer_free(optimizer);
    CM_FREE(params);
    module_free(model);

    autograd_shutdown();
}

int main(void) {
    training_example();
    return 0;
}
