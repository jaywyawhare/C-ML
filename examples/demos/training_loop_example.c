#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static void training_example(void) {
    cml_init();
    cml_seed(42);

    int input_size  = 10;
    int hidden_size = 20;
    int output_size = 1;

    Sequential* seq   = nn_sequential();
    DeviceType device = cml_get_default_device();
    DType dtype       = cml_get_default_dtype();

    seq           = cml_nn_sequential_add(seq,
                                          (Module*)nn_linear(input_size, hidden_size, dtype, device, true));
    seq           = cml_nn_sequential_add(seq, (Module*)nn_relu(false));
    seq           = cml_nn_sequential_add(seq,
                                          (Module*)nn_linear(hidden_size, output_size, dtype, device, true));
    seq           = cml_nn_sequential_add(seq, (Module*)nn_sigmoid());
    Module* model = (Module*)seq;

    if (!model) {
        LOG_ERROR("Failed to create model");
        return;
    }
    module_set_training(model, true);

    cml_summary(model);
    training_metrics_register_model(model);

    Optimizer* optimizer = NULL;
    Tensor* inputs       = NULL;
    Tensor* targets      = NULL;

    optimizer = optim_adam_for_model(model, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

    if (!optimizer) {
        LOG_ERROR("Failed to create optimizer");
        return;
    }

    printf("Created %s optimizer\n", optimizer_get_name(optimizer));

    int batch_size  = 32;
    int num_samples = 100;

    inputs = tensor_zeros_2d(num_samples, input_size);
    if (!inputs) {
        LOG_ERROR("Failed to create input tensor");
        return;
    }

    targets = tensor_zeros_2d(num_samples, output_size);
    if (!targets) {
        LOG_ERROR("Failed to create target tensor");
        tensor_free(inputs);
        return;
    }

    float* input_data  = (float*)tensor_data_ptr(inputs);
    float* target_data = (float*)tensor_data_ptr(targets);

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < input_size; j++) {
            input_data[i * input_size + j] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        }

        float sum = 0.0f;
        for (int j = 0; j < 3 && j < input_size; j++) {
            sum += input_data[i * input_size + j];
        }
        target_data[i] = (sum > 0.0f) ? 1.0f : 0.0f;
    }

    int num_epochs        = 100;
    float best_loss       = INFINITY;
    int patience          = 10;
    int no_improve_epochs = 0;
    float improvement_tol = 1e-5f;
    int lr_step_size      = 20;
    float lr_gamma        = 0.5f;

    training_metrics_set_expected_epochs((size_t)num_epochs);

    printf("\nStarting training for %d epochs...\n\n", num_epochs);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        float epoch_acc  = 0.0f;
        int num_batches  = num_samples / batch_size;
        if (num_batches == 0)
            num_batches = 1;

        for (int batch = 0; batch < num_batches; batch++) {
            optimizer_zero_grad(optimizer);

            int batch_start = (batch * batch_size) % num_samples;
            int batch_end   = batch_start + batch_size;
            if (batch_end > num_samples)
                batch_end = num_samples;
            int current_batch_size = batch_end - batch_start;

            Tensor* batch_inputs  = tensor_zeros_2d(current_batch_size, input_size);
            Tensor* batch_targets = tensor_zeros_2d(current_batch_size, output_size);

            float* batch_in_data  = (float*)tensor_data_ptr(batch_inputs);
            float* batch_tgt_data = (float*)tensor_data_ptr(batch_targets);
            float* all_in_data    = (float*)tensor_data_ptr(inputs);
            float* all_tgt_data   = (float*)tensor_data_ptr(targets);

            for (int i = 0; i < current_batch_size; i++) {
                int src_idx = batch_start + i;
                memcpy(batch_in_data + i * input_size, all_in_data + src_idx * input_size,
                       (size_t)input_size * sizeof(float));
                memcpy(batch_tgt_data + i * output_size, all_tgt_data + src_idx * output_size,
                       (size_t)output_size * sizeof(float));
            }

            Tensor* outputs = module_forward(model, batch_inputs);

            if (!outputs) {
                LOG_ERROR("Forward pass failed");
                tensor_free(batch_inputs);
                tensor_free(batch_targets);
                continue;
            }

            Tensor* loss = tensor_mse_loss(outputs, batch_targets);

            if (!loss) {
                LOG_ERROR("Loss computation failed");
                tensor_free(outputs);
                tensor_free(batch_inputs);
                tensor_free(batch_targets);
                continue;
            }

            float* out_data = (float*)tensor_data_ptr(outputs);
            int correct     = 0;
            for (int i = 0; i < current_batch_size; i++) {
                float pred = out_data[i];
                float tgt  = batch_tgt_data[i];
                if ((pred > 0.5f && tgt > 0.5f) || (pred <= 0.5f && tgt <= 0.5f)) {
                    correct++;
                }
            }
            epoch_acc += (float)correct / (float)current_batch_size;

            if (batch == 0 && epoch == 0) {
                float p0 = tensor_get_float(outputs, 0);
                float p1 = tensor_get_float(outputs, 1);
                float p2 = tensor_get_float(outputs, 2);
                float t0 = tensor_get_float(batch_targets, 0);
                float t1 = tensor_get_float(batch_targets, 1);
                float t2 = tensor_get_float(batch_targets, 2);
                printf(
                    "    Sample predictions: [%.3f, %.3f, %.3f] vs targets: [%.3f, %.3f, %.3f]\n",
                    (double)p0, (double)p1, (double)p2, (double)t0, (double)t1, (double)t2);
            }

            tensor_backward(loss, NULL, false, false);

            if (batch == 0 && epoch == 0) {
                autograd_export_json(loss, "graph.json");

                if (loss->ir_context) {
                    char* unopt = cml_ir_export_kernel_analysis(loss->ir_context, false);
                    cml_ir_optimize(loss->ir_context);
                    char* opt = cml_ir_export_kernel_analysis(loss->ir_context, true);

                    if (unopt && opt) {
                        FILE* f = fopen("kernels.json", "w");
                        if (f) {
                            fprintf(f, "{\"unoptimized\":%s,\"optimized\":%s}", unopt, opt);
                            fclose(f);
                        }
                    }
                    if (unopt)
                        free(unopt);
                    if (opt)
                        free(opt);
                }
            }

            float* loss_data = (float*)tensor_data_ptr(loss);
            if (loss_data) {
                epoch_loss += loss_data[0];
            }

            optimizer_step(optimizer);
            cml_reset_ir_context();

            if (loss)
                tensor_free(loss);
            if (outputs)
                tensor_free(outputs);
            tensor_free(batch_inputs);
            tensor_free(batch_targets);
        }

        epoch_loss /= (float)num_batches;
        epoch_acc /= (float)num_batches;

        training_metrics_auto_capture_train_accuracy(epoch_acc);

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            printf("Epoch %d/%d - Loss: %.4f - Acc: %.2f%%\n", epoch + 1, num_epochs,
                   (double)epoch_loss, (double)(epoch_acc * 100.0f));
        }

        if (epoch_loss + improvement_tol < best_loss) {
            best_loss         = epoch_loss;
            no_improve_epochs = 0;
            printf("  (New best model! Loss %.6f)\n", (double)best_loss);
        } else {
            no_improve_epochs++;
            if (no_improve_epochs >= patience) {
                printf("  Early stopping triggered (no improvement for %d epochs)\n", patience);
                break;
            }
        }

        if ((epoch + 1) % lr_step_size == 0) {
            float current_lr = optimizer_get_group_lr(optimizer, 0);
            float new_lr     = current_lr * lr_gamma;
            optimizer_set_lr(optimizer, new_lr);
            printf("  LR decayed to %.6f\n", (double)new_lr);
        }
    }

    printf("\nTraining completed!\n");

    training_metrics_complete_epoch();

    Tensor* eval_out = module_forward(model, inputs);
    if (eval_out) {
        printf("\nEval snapshot (first 5):\n");
        size_t limit = eval_out->numel;
        if (limit > targets->numel)
            limit = targets->numel;
        if (limit > 5)
            limit = 5;
        for (size_t i = 0; i < limit; i++) {
            printf("  Sample %zu: pred=%.4f  target=%.4f\n", i,
                   (double)tensor_get_float(eval_out, i), (double)tensor_get_float(targets, i));
        }
        tensor_free(eval_out);
    }

    tensor_free(targets);
    tensor_free(inputs);
    cml_cleanup();
}

int main(void) {
    training_example();
    return 0;
}
