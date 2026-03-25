#define _POSIX_C_SOURCE 199309L

#include "cml.h"
#include "core/mlperf_logging.h"
#include "zoo/resnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct {
    const char* data_dir;
    int batch_size;
    int epochs;
    float lr;
    float momentum;
    float weight_decay;
    float target_accuracy;
} MLPerfConfig;

static MLPerfConfig parse_args(int argc, char** argv) {
    MLPerfConfig cfg = {
        .data_dir = "/data/imagenet",
        .batch_size = 256,
        .epochs = 90,
        .lr = 0.1f,
        .momentum = 0.9f,
        .weight_decay = 1e-4f,
        .target_accuracy = 0.759f,
    };

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--data-dir=", 11) == 0)
            cfg.data_dir = argv[i] + 11;
        else if (strncmp(argv[i], "--batch-size=", 13) == 0)
            cfg.batch_size = atoi(argv[i] + 13);
        else if (strncmp(argv[i], "--epochs=", 9) == 0)
            cfg.epochs = atoi(argv[i] + 9);
        else if (strncmp(argv[i], "--lr=", 5) == 0)
            cfg.lr = (float)atof(argv[i] + 5);
        else if (strncmp(argv[i], "--momentum=", 11) == 0)
            cfg.momentum = (float)atof(argv[i] + 11);
        else if (strncmp(argv[i], "--weight-decay=", 15) == 0)
            cfg.weight_decay = (float)atof(argv[i] + 15);
        else if (strncmp(argv[i], "--target-accuracy=", 18) == 0)
            cfg.target_accuracy = (float)atof(argv[i] + 18);
    }
    return cfg;
}

static float get_lr(float base_lr, int epoch, int warmup_epochs) {
    if (epoch < warmup_epochs)
        return base_lr * ((float)(epoch + 1) / warmup_epochs);

    if (epoch < 30) return base_lr;
    if (epoch < 60) return base_lr * 0.1f;
    if (epoch < 80) return base_lr * 0.01f;
    return base_lr * 0.001f;
}

static void random_crop_flip(float* img, int h, int w, int c) {
    int crop_h = (int)(h * 0.875f);
    int crop_w = (int)(w * 0.875f);
    int off_y = rand() % (h - crop_h + 1);
    int off_x = rand() % (w - crop_w + 1);

    float* tmp = malloc(sizeof(float) * crop_h * crop_w * c);
    if (!tmp) return;

    for (int y = 0; y < crop_h; y++)
        for (int x = 0; x < crop_w; x++)
            for (int ch = 0; ch < c; ch++)
                tmp[(y * crop_w + x) * c + ch] =
                    img[((off_y + y) * w + (off_x + x)) * c + ch];

    bool flip = (rand() % 2) == 0;
    if (flip) {
        for (int y = 0; y < crop_h; y++)
            for (int x = 0; x < crop_w / 2; x++)
                for (int ch = 0; ch < c; ch++) {
                    int l = (y * crop_w + x) * c + ch;
                    int r = (y * crop_w + (crop_w - 1 - x)) * c + ch;
                    float t = tmp[l];
                    tmp[l] = tmp[r];
                    tmp[r] = t;
                }
    }

    memcpy(img, tmp, sizeof(float) * crop_h * crop_w * c);
    free(tmp);
}

static int compute_topk(Tensor* logits, int* labels, int batch_size, int num_classes, int k) {
    int correct = 0;
    float* data = (float*)tensor_data_ptr(logits);
    if (!data) return 0;

    for (int b = 0; b < batch_size; b++) {
        float* row = data + b * num_classes;
        int target = labels[b];

        int top_indices[5] = {0};
        float top_values[5];
        for (int i = 0; i < k; i++) top_values[i] = -1e30f;

        for (int c = 0; c < num_classes; c++) {
            int min_idx = 0;
            for (int i = 1; i < k; i++)
                if (top_values[i] < top_values[min_idx]) min_idx = i;
            if (row[c] > top_values[min_idx]) {
                top_values[min_idx] = row[c];
                top_indices[min_idx] = c;
            }
        }

        for (int i = 0; i < k; i++)
            if (top_indices[i] == target) { correct++; break; }
    }
    return correct;
}

static double time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char** argv) {
    cml_init();
    srand((unsigned)time(NULL));

    MLPerfConfig cfg = parse_args(argc, argv);
    int num_classes = 1000;
    int warmup_epochs = 5;
    int img_h = 224, img_w = 224, img_c = 3;

    mlperf_log_start("resnet50");
    mlperf_log_event("benchmark", "resnet50_v1.5");
    mlperf_log_metric("batch_size", cfg.batch_size);
    mlperf_log_metric("learning_rate", cfg.lr);
    mlperf_log_metric("target_accuracy", cfg.target_accuracy);

    Module* model = cml_zoo_resnet50_create(num_classes, DTYPE_FLOAT32, DEVICE_CPU);
    if (!model) {
        fprintf(stderr, "Failed to create ResNet-50 model\n");
        return 1;
    }

    Optimizer* opt = cml_optim_sgd_for_model(model, cfg.lr, cfg.momentum, cfg.weight_decay);
    if (!opt) {
        fprintf(stderr, "Failed to create optimizer\n");
        module_free(model);
        return 1;
    }

    mlperf_log_event("model_created", "resnet50");

    int samples_per_epoch = 1281167;
    int steps_per_epoch = samples_per_epoch / cfg.batch_size;

    int input_shape[] = { cfg.batch_size, img_c, img_h, img_w };

    float* batch_data = malloc(sizeof(float) * cfg.batch_size * img_c * img_h * img_w);
    int* batch_labels = malloc(sizeof(int) * cfg.batch_size);

    if (!batch_data || !batch_labels) {
        fprintf(stderr, "Failed to allocate batch buffers\n");
        free(batch_data);
        free(batch_labels);
        optimizer_free(opt);
        module_free(model);
        return 1;
    }

    bool converged = false;
    double train_start = time_seconds();

    for (int epoch = 0; epoch < cfg.epochs && !converged; epoch++) {
        double epoch_start = time_seconds();
        float current_lr = get_lr(cfg.lr, epoch, warmup_epochs);
        mlperf_log_metric("current_lr", current_lr);

        float epoch_loss = 0.0f;
        int epoch_samples = 0;

        for (int step = 0; step < steps_per_epoch; step++) {
            for (int b = 0; b < cfg.batch_size; b++) {
                for (int i = 0; i < img_c * img_h * img_w; i++)
                    batch_data[b * img_c * img_h * img_w + i] =
                        (float)rand() / RAND_MAX;
                random_crop_flip(batch_data + b * img_c * img_h * img_w,
                                 img_h, img_w, img_c);
                batch_labels[b] = rand() % num_classes;
            }

            Tensor* input = cml_tensor(batch_data, input_shape, 4, NULL);
            Tensor* logits = module_forward(model, input);

            if (!logits) {
                fprintf(stderr, "Forward pass failed at step %d\n", step);
                continue;
            }

            float* label_floats = malloc(sizeof(float) * cfg.batch_size * num_classes);
            if (!label_floats) continue;
            memset(label_floats, 0, sizeof(float) * cfg.batch_size * num_classes);
            for (int b = 0; b < cfg.batch_size; b++)
                label_floats[b * num_classes + batch_labels[b]] = 1.0f;

            int target_shape[] = { cfg.batch_size, num_classes };
            Tensor* target = cml_tensor(label_floats, target_shape, 2, NULL);
            Tensor* loss = cml_nn_cross_entropy_loss(logits, target);

            cml_optim_zero_grad(opt);
            cml_backward(loss, NULL, false, false);
            cml_optim_step(opt);

            float loss_val = tensor_get_float(loss, 0);
            epoch_loss += loss_val;
            epoch_samples += cfg.batch_size;

            if (step % 100 == 0) {
                double throughput = epoch_samples / (time_seconds() - epoch_start);
                printf("Epoch %d Step %d/%d  Loss: %.4f  Throughput: %.1f samples/sec\n",
                       epoch + 1, step, steps_per_epoch, loss_val, throughput);
            }

            free(label_floats);
            cml_reset_ir_context();
        }

        double epoch_time = time_seconds() - epoch_start;
        float avg_loss = epoch_loss / steps_per_epoch;
        double throughput = epoch_samples / epoch_time;

        mlperf_log_metric("epoch_loss", avg_loss);
        mlperf_log_metric("samples_per_second", throughput);

        int eval_correct_top1 = 0;
        int eval_correct_top5 = 0;
        int eval_total = 0;
        int eval_samples = 50000;
        int eval_steps = eval_samples / cfg.batch_size;

        for (int step = 0; step < eval_steps; step++) {
            for (int b = 0; b < cfg.batch_size; b++) {
                for (int i = 0; i < img_c * img_h * img_w; i++)
                    batch_data[b * img_c * img_h * img_w + i] =
                        (float)rand() / RAND_MAX;
                batch_labels[b] = rand() % num_classes;
            }

            Tensor* input = cml_tensor(batch_data, input_shape, 4, NULL);
            Tensor* logits = module_forward(model, input);
            if (!logits) continue;

            eval_correct_top1 += compute_topk(logits, batch_labels, cfg.batch_size, num_classes, 1);
            eval_correct_top5 += compute_topk(logits, batch_labels, cfg.batch_size, num_classes, 5);
            eval_total += cfg.batch_size;
        }

        float top1_acc = eval_total > 0 ? (float)eval_correct_top1 / eval_total : 0.0f;
        float top5_acc = eval_total > 0 ? (float)eval_correct_top5 / eval_total : 0.0f;

        mlperf_log_metric("eval_top1_accuracy", top1_acc);
        mlperf_log_metric("eval_top5_accuracy", top5_acc);

        printf("Epoch %d/%d  Loss: %.4f  Top-1: %.4f  Top-5: %.4f  "
               "Time: %.1fs  Throughput: %.1f img/s\n",
               epoch + 1, cfg.epochs, avg_loss, top1_acc, top5_acc,
               epoch_time, throughput);

        if (top1_acc >= cfg.target_accuracy) {
            double total_time = time_seconds() - train_start;
            printf("Target accuracy %.3f reached at epoch %d (%.1f seconds)\n",
                   cfg.target_accuracy, epoch + 1, total_time);
            mlperf_log_metric("time_to_accuracy", total_time);
            converged = true;
        }
    }

    double total_time = time_seconds() - train_start;
    mlperf_log_metric("total_training_time", total_time);
    mlperf_log_end("resnet50", converged ? "success" : "aborted");

    free(batch_data);
    free(batch_labels);
    optimizer_free(opt);
    module_free(model);

    printf("Training complete. Total time: %.1f seconds\n", total_time);
    return converged ? 0 : 1;
}
