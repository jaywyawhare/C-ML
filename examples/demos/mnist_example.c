/**
 * @file mnist_example.c
 * @brief MNIST digit classification example using C-ML
 *
 * This example demonstrates training a neural network on the MNIST dataset.
 *
 * To run this example, you need to download the MNIST dataset:
 *   wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
 *   wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
 *   wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
 *   wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
 *   gunzip *.gz
 *
 * Then run:
 *   ./mnist_example [data_dir]
 */

#include "cml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

// Wall-clock time helper (clock() measures CPU time which is wrong for multi-threaded code)
static double get_time_sec(void) {
#if defined(__linux__) || defined(__APPLE__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

// IDX file format helpers
static uint32_t read_uint32_be(FILE* f) {
    uint8_t bytes[4];
    if (fread(bytes, 1, 4, f) != 4)
        return 0;
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) |
           (uint32_t)bytes[3];
}

// Load MNIST images from IDX file
static float* load_mnist_images(const char* filename, int* num_images, int* rows, int* cols) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error: Cannot open %s\n", filename);
        return NULL;
    }

    uint32_t magic = read_uint32_be(f);
    if (magic != 2051) {
        printf("Error: Invalid magic number %u (expected 2051)\n", magic);
        fclose(f);
        return NULL;
    }

    *num_images = (int)read_uint32_be(f);
    *rows       = (int)read_uint32_be(f);
    *cols       = (int)read_uint32_be(f);

    printf("Loading %d images (%dx%d)...\n", *num_images, *rows, *cols);

    int image_size = (*rows) * (*cols);
    float* data    = (float*)malloc(sizeof(float) * (*num_images) * image_size);
    if (!data) {
        fclose(f);
        return NULL;
    }

    // Read and normalize pixel values
    uint8_t* buffer = (uint8_t*)malloc(image_size);
    for (int i = 0; i < *num_images; i++) {
        if (fread(buffer, 1, image_size, f) != (size_t)image_size) {
            printf("Error: Failed to read image %d\n", i);
            free(data);
            free(buffer);
            fclose(f);
            return NULL;
        }
        // Normalize to [0, 1]
        for (int j = 0; j < image_size; j++) {
            data[i * image_size + j] = buffer[j] / 255.0f;
        }
    }

    free(buffer);
    fclose(f);
    return data;
}

// Load MNIST labels from IDX file as one-hot encoded (for MSE loss - faster)
static float* load_mnist_labels(const char* filename, int* num_labels) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error: Cannot open %s\n", filename);
        return NULL;
    }

    uint32_t magic = read_uint32_be(f);
    if (magic != 2049) {
        printf("Error: Invalid magic number %u (expected 2049)\n", magic);
        fclose(f);
        return NULL;
    }

    *num_labels = (int)read_uint32_be(f);
    printf("Loading %d labels...\n", *num_labels);

    // One-hot encode for MSE loss (faster than cross-entropy which breaks lazy eval)
    float* data = (float*)calloc((*num_labels) * 10, sizeof(float));
    if (!data) {
        fclose(f);
        return NULL;
    }

    for (int i = 0; i < *num_labels; i++) {
        uint8_t label;
        if (fread(&label, 1, 1, f) != 1) {
            printf("Error: Failed to read label %d\n", i);
            free(data);
            fclose(f);
            return NULL;
        }
        if (label < 10) {
            data[i * 10 + label] = 1.0f;
        }
    }

    fclose(f);
    return data;
}

// Calculate accuracy: argmax of predictions vs one-hot targets
static float calculate_accuracy(Tensor* predictions, Tensor* targets, int num_samples) {
    float* pred_data   = (float*)tensor_data_ptr(predictions);
    float* target_data = (float*)tensor_data_ptr(targets); // One-hot [N, 10]
    if (!pred_data || !target_data)
        return 0.0f;

    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        int pred_class = 0, target_class = 0;
        float pred_max   = pred_data[i * 10];
        float target_max = target_data[i * 10];

        for (int j = 1; j < 10; j++) {
            if (pred_data[i * 10 + j] > pred_max) {
                pred_max   = pred_data[i * 10 + j];
                pred_class = j;
            }
            if (target_data[i * 10 + j] > target_max) {
                target_max   = target_data[i * 10 + j];
                target_class = j;
            }
        }

        if (pred_class == target_class) {
            correct++;
        }
    }

    return (float)correct / num_samples;
}

int main(int argc, char** argv) {
    const char* data_dir = argc > 1 ? argv[1] : ".";

    // Force line buffering for immediate output
    setvbuf(stdout, NULL, _IOLBF, 0);

    printf("=== MNIST Classification with C-ML ===\n\n");

    // Build file paths
    char train_images_path[512], train_labels_path[512];
    char test_images_path[512], test_labels_path[512];
    snprintf(train_images_path, sizeof(train_images_path), "%s/train-images-idx3-ubyte", data_dir);
    snprintf(train_labels_path, sizeof(train_labels_path), "%s/train-labels-idx1-ubyte", data_dir);
    snprintf(test_images_path, sizeof(test_images_path), "%s/t10k-images-idx3-ubyte", data_dir);
    snprintf(test_labels_path, sizeof(test_labels_path), "%s/t10k-labels-idx1-ubyte", data_dir);

    // Load training data
    printf("Loading training data...\n");
    int num_train, rows, cols;
    float* train_images = load_mnist_images(train_images_path, &num_train, &rows, &cols);
    if (!train_images) {
        printf("\nPlease download MNIST dataset:\n");
        printf("  wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\n");
        printf("  wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\n");
        printf("  wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\n");
        printf("  wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\n");
        printf("  gunzip *.gz\n");
        return 1;
    }

    int num_train_labels;
    float* train_labels = load_mnist_labels(train_labels_path, &num_train_labels);
    if (!train_labels) {
        free(train_images);
        return 1;
    }

    // Load test data
    printf("\nLoading test data...\n");
    int num_test, test_rows, test_cols;
    float* test_images = load_mnist_images(test_images_path, &num_test, &test_rows, &test_cols);
    int num_test_labels;
    float* test_labels = test_images ? load_mnist_labels(test_labels_path, &num_test_labels) : NULL;

    // Use a subset of training data (increase for better accuracy)
    int train_subset = 10000; // Use 10k samples (increase to 60000 for full training)
    if (train_subset > num_train)
        train_subset = num_train;
    printf("\nUsing %d training samples (out of %d)\n", train_subset, num_train);

    // Initialize C-ML
    cml_init();
    cml_seed(42);

    // Create tensors
    int input_size  = rows * cols; // 784 for MNIST
    int num_classes = 10;

    int x_shape[]       = {train_subset, input_size};
    int y_shape[]       = {train_subset, num_classes}; // 2D: one-hot for MSE
    TensorConfig config = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    Tensor* X_train = tensor_from_data(train_images, x_shape, 2, &config);
    Tensor* y_train = tensor_from_data(train_labels, y_shape, 2, &config); // 2D one-hot

    printf("\nX_train shape: [%d, %d]\n", x_shape[0], x_shape[1]);
    printf("y_train shape: [%d, %d] (one-hot)\n", y_shape[0], y_shape[1]);

    // Build model: Simple MLP for MNIST
    // Input (784) -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(10) -> Sigmoid
    // Using sigmoid + MSE (fully IR-based, much faster than cross-entropy)
    printf("\nBuilding model...\n");
    Sequential* model = cml_nn_sequential();
    DeviceType device = cml_get_default_device();
    DType dtype       = cml_get_default_dtype();

    model =
        cml_nn_sequential_add(model, (Module*)cml_nn_linear(input_size, 128, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(128, 64, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model =
        cml_nn_sequential_add(model, (Module*)cml_nn_linear(64, num_classes, dtype, device, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_sigmoid()); // Sigmoid for MSE

    cml_summary((Module*)model);
    cml_nn_module_set_training((Module*)model, true);

    // Create optimizer
    Optimizer* optimizer =
        cml_optim_adam_for_model((Module*)model, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);
    printf("\nOptimizer: Adam (lr=0.001)\n");

    // Training loop
    int num_epochs  = 10; // Train for 10 epochs
    int batch_size  = 64;
    int num_batches = (train_subset + batch_size - 1) / batch_size;

    printf("\nTraining for %d epochs (batch_size=%d, batches=%d)...\n\n", num_epochs, batch_size,
           num_batches);

    // Set up training metrics for visualization
    training_metrics_set_expected_epochs(num_epochs);
    training_metrics_register_model((Module*)model);

    double total_start = get_time_sec();

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double epoch_start = get_time_sec();
        float epoch_loss   = 0.0f;
        int total_correct  = 0;

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx   = start_idx + batch_size;
            if (end_idx > train_subset)
                end_idx = train_subset;
            int current_batch_size = end_idx - start_idx;

            // Create batch tensors (slice from full data)
            int batch_x_shape[] = {current_batch_size, input_size};
            int batch_y_shape[] = {current_batch_size, num_classes}; // 2D one-hot

            Tensor* X_batch =
                tensor_from_data(&train_images[start_idx * input_size], batch_x_shape, 2, &config);
            Tensor* y_batch =
                tensor_from_data(&train_labels[start_idx * num_classes], batch_y_shape, 2, &config);

            // Zero gradients
            if (epoch == 0 && batch % 10 == 0) {
                printf("\r  Batch %d/%d", batch, num_batches);
                fflush(stdout);
            }
            cml_optim_zero_grad(optimizer);

            // Forward pass
            Tensor* outputs = cml_nn_module_forward((Module*)model, X_batch);
            if (!outputs) {
                printf("Error: Forward pass failed at epoch %d, batch %d\n", epoch, batch);
                break;
            }

            // Compute MSE loss (fully IR-based, no eager execution)
            Tensor* loss = cml_nn_mse_loss(outputs, y_batch);
            if (!loss) {
                printf("Error: Loss computation failed\n");
                break;
            }

            float batch_loss = tensor_get_float(loss, 0);
            epoch_loss += batch_loss;

            // Calculate batch accuracy
            float batch_acc = calculate_accuracy(outputs, y_batch, current_batch_size);
            total_correct += (int)(batch_acc * current_batch_size);

            // Backward pass
            cml_backward(loss, NULL, false, false);

            // Export graph and kernels on first batch for visualization
            if (epoch == 0 && batch == 0) {
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

            // Optimizer step
            cml_optim_step(optimizer);

            // Free batch tensors and intermediate results
            tensor_free(loss);
            tensor_free(outputs);
            tensor_free(X_batch);
            tensor_free(y_batch);

            // Reset IR context every batch to prevent memory growth
            // This trades JIT compilation speed for memory stability
            cml_reset_ir_context();
        }

        float avg_loss    = epoch_loss / num_batches;
        float train_acc   = (float)total_correct / train_subset * 100.0f;
        double epoch_time = get_time_sec() - epoch_start;

        // Capture metrics for visualization (accuracy as 0-1 fraction)
        training_metrics_auto_capture_train_accuracy(train_acc / 100.0f);
        training_metrics_complete_epoch();

        printf("Epoch %2d/%d - Loss: %.4f, Accuracy: %.2f%%, Time: %.2fs\n", epoch + 1, num_epochs,
               (double)avg_loss, (double)train_acc, epoch_time);
    }

    double total_time = get_time_sec() - total_start;
    printf("\nTotal training time: %.2fs (%.2f samples/sec)\n", total_time,
           (num_epochs * train_subset) / total_time);

    // Test evaluation
    if (test_images && test_labels) {
        printf("\n=== Testing on %d samples ===\n", num_test);

        int test_x_shape[] = {num_test, input_size};
        int test_y_shape[] = {num_test, num_classes}; // 2D one-hot

        Tensor* X_test = tensor_from_data(test_images, test_x_shape, 2, &config);
        Tensor* y_test = tensor_from_data(test_labels, test_y_shape, 2, &config);

        cml_nn_module_set_training((Module*)model, false);
        Tensor* test_outputs = cml_nn_module_forward((Module*)model, X_test);

        if (test_outputs) {
            float test_acc = calculate_accuracy(test_outputs, y_test, num_test);
            printf("Test Accuracy: %.2f%%\n", (double)(test_acc * 100.0f));
        }

        tensor_free(X_test);
        tensor_free(y_test);
    }

    // Cleanup
    printf("\n=== Done ===\n");
    free(train_images);
    free(train_labels);
    if (test_images)
        free(test_images);
    if (test_labels)
        free(test_labels);

    tensor_free(X_train);
    tensor_free(y_train);
    cml_cleanup();

    return 0;
}
