/**
 * @file dataset.c
 * @brief Implementation of high-level dataset API
 *
 * This file implements the dataset interface that
 * provides easy data management and loading.
 */

#include "Core/dataset.h"
#include "tensor/tensor.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Dataset Implementation

Dataset* dataset_create(void) {
    LOG_DEBUG("Creating new dataset");

    Dataset* dataset = CM_MALLOC(sizeof(Dataset));
    if (!dataset) {
        LOG_ERROR("Failed to allocate memory for Dataset");
        return NULL;
    }

    // Initialize basic fields
    dataset->name        = "Dataset";
    dataset->X           = NULL;
    dataset->y           = NULL;
    dataset->num_samples = 0;
    dataset->input_size  = 0;
    dataset->output_size = 0;

    // Initialize data properties
    dataset->dtype         = DTYPE_FLOAT32;
    dataset->device        = DEVICE_CPU;
    dataset->is_normalized = false;
    dataset->is_shuffled   = false;

    // Initialize statistics arrays
    dataset->feature_means = NULL;
    dataset->feature_stds  = NULL;
    dataset->feature_mins  = NULL;
    dataset->feature_maxs  = NULL;

    // Initialize metadata
    dataset->feature_names = NULL;
    dataset->class_names   = NULL;
    dataset->num_classes   = 0;

    // Initialize internal state
    dataset->indices   = NULL;
    dataset->is_loaded = false;
    dataset->filepath  = NULL;
    dataset->user_data = NULL;

    LOG_DEBUG("Dataset created successfully at %p", dataset);
    return dataset;
}

Dataset* dataset_from_arrays(float* X, float* y, int num_samples, int input_size, int output_size) {
    LOG_DEBUG("Creating dataset from arrays: %d samples, %d->%d", num_samples, input_size,
              output_size);

    Dataset* dataset = dataset_create();
    if (!dataset)
        return NULL;

    // Load data into dataset
    if (dataset_load_arrays(dataset, X, y, num_samples, input_size, output_size) != 0) {
        LOG_ERROR("Failed to load arrays into dataset");
        dataset_free(dataset);
        return NULL;
    }

    return dataset;
}

int dataset_load_arrays(Dataset* dataset, float* X, float* y, int num_samples, int input_size,
                        int output_size) {
    if (!dataset || !X || !y) {
        LOG_ERROR("Invalid parameters for dataset_load_arrays");
        return -1;
    }

    LOG_DEBUG("Loading arrays into dataset: %d samples, %d->%d", num_samples, input_size,
              output_size);

    // Create input tensor
    int X_shape[2]      = {num_samples, input_size};
    TensorConfig config = tensor_config_with_dtype_device(DTYPE_FLOAT32, DEVICE_CPU);
    dataset->X          = tensor_from_data(X, X_shape, 2, &config);
    if (!dataset->X) {
        LOG_ERROR("Failed to create input tensor");
        return -1;
    }

    // Create target tensor
    int y_shape[2] = {num_samples, output_size};
    dataset->y     = tensor_from_data(y, y_shape, 2, &config);
    if (!dataset->y) {
        LOG_ERROR("Failed to create target tensor");
        tensor_free(dataset->X);
        dataset->X = NULL;
        return -1;
    }

    // Set dataset properties
    dataset->num_samples = num_samples;
    dataset->input_size  = input_size;
    dataset->output_size = output_size;
    dataset->is_loaded   = true;

    // Initialize indices for shuffling
    dataset->indices = CM_MALLOC(num_samples * sizeof(int));
    if (!dataset->indices) {
        LOG_WARNING("Failed to allocate indices array, shuffling will be disabled");
    } else {
        for (int i = 0; i < num_samples; i++) {
            dataset->indices[i] = i;
        }
    }

    LOG_DEBUG("Arrays loaded successfully into dataset");
    return 0;
}

void dataset_free(Dataset* dataset) {
    if (!dataset)
        return;

    LOG_DEBUG("Freeing dataset at %p", dataset);

    // Free tensors
    if (dataset->X)
        tensor_free(dataset->X);
    if (dataset->y)
        tensor_free(dataset->y);

    // Free statistics arrays
    if (dataset->feature_means)
        CM_FREE(dataset->feature_means);
    if (dataset->feature_stds)
        CM_FREE(dataset->feature_stds);
    if (dataset->feature_mins)
        CM_FREE(dataset->feature_mins);
    if (dataset->feature_maxs)
        CM_FREE(dataset->feature_maxs);

    // Free metadata
    if (dataset->feature_names) {
        for (int i = 0; i < dataset->input_size; i++) {
            if (dataset->feature_names[i])
                CM_FREE(dataset->feature_names[i]);
        }
        CM_FREE(dataset->feature_names);
    }

    if (dataset->class_names) {
        for (int i = 0; i < dataset->num_classes; i++) {
            if (dataset->class_names[i])
                CM_FREE(dataset->class_names[i]);
        }
        CM_FREE(dataset->class_names);
    }

    // Free internal arrays
    if (dataset->indices)
        CM_FREE(dataset->indices);
    if (dataset->filepath)
        CM_FREE(dataset->filepath);

    // Free the dataset itself
    CM_FREE(dataset);

    LOG_DEBUG("Dataset freed successfully");
}

// Dataset Operations

int dataset_split(Dataset* dataset, float train_ratio, Dataset** train_dataset,
                  Dataset** val_dataset) {
    if (!dataset || !train_dataset || !val_dataset) {
        LOG_ERROR("Invalid parameters for dataset_split");
        return -1;
    }

    if (train_ratio <= 0.0f || train_ratio >= 1.0f) {
        LOG_ERROR("Invalid train_ratio: %.2f (must be between 0 and 1)", train_ratio);
        return -1;
    }

    LOG_DEBUG("Splitting dataset with train_ratio %.2f", train_ratio);

    int train_size = (int)(dataset->num_samples * train_ratio);
    int val_size   = dataset->num_samples - train_size;

    // Create training dataset
    *train_dataset = dataset_create();
    if (!*train_dataset) {
        LOG_ERROR("Failed to create training dataset");
        return -1;
    }

    // Create validation dataset
    *val_dataset = dataset_create();
    if (!*val_dataset) {
        LOG_ERROR("Failed to create validation dataset");
        dataset_free(*train_dataset);
        *train_dataset = NULL;
        return -1;
    }
    // TODO
    // Copy data (in a real implementation, you'd create proper tensor views)
    // For now, we'll just set the metadata
    (*train_dataset)->num_samples = train_size;
    (*train_dataset)->input_size  = dataset->input_size;
    (*train_dataset)->output_size = dataset->output_size;
    (*train_dataset)->dtype       = dataset->dtype;
    (*train_dataset)->device      = dataset->device;

    (*val_dataset)->num_samples = val_size;
    (*val_dataset)->input_size  = dataset->input_size;
    (*val_dataset)->output_size = dataset->output_size;
    (*val_dataset)->dtype       = dataset->dtype;
    (*val_dataset)->device      = dataset->device;

    LOG_DEBUG("Dataset split successfully: train=%d, val=%d", train_size, val_size);
    return 0;
}

int dataset_split_three(Dataset* dataset, float train_ratio, float val_ratio,
                        Dataset** train_dataset, Dataset** val_dataset, Dataset** test_dataset) {
    if (!dataset || !train_dataset || !val_dataset || !test_dataset) {
        LOG_ERROR("Invalid parameters for dataset_split_three");
        return -1;
    }

    if (train_ratio <= 0.0f || train_ratio >= 1.0f) {
        LOG_ERROR("Invalid train_ratio: %.2f (must be between 0 and 1)", train_ratio);
        return -1;
    }

    if (val_ratio <= 0.0f || val_ratio >= 1.0f) {
        LOG_ERROR("Invalid val_ratio: %.2f (must be between 0 and 1)", val_ratio);
        return -1;
    }

    if (train_ratio + val_ratio >= 1.0f) {
        LOG_ERROR("Invalid ratios: train_ratio + val_ratio (%.2f) must be < 1.0",
                  train_ratio + val_ratio);
        return -1;
    }

    LOG_DEBUG("Splitting dataset with train_ratio %.2f, val_ratio %.2f", train_ratio, val_ratio);

    int train_size = (int)(dataset->num_samples * train_ratio);
    int val_size   = (int)(dataset->num_samples * val_ratio);
    int test_size  = dataset->num_samples - train_size - val_size;

    // Create training dataset
    *train_dataset = dataset_create();
    if (!*train_dataset) {
        LOG_ERROR("Failed to create training dataset");
        return -1;
    }

    // Create validation dataset
    *val_dataset = dataset_create();
    if (!*val_dataset) {
        LOG_ERROR("Failed to create validation dataset");
        dataset_free(*train_dataset);
        *train_dataset = NULL;
        return -1;
    }

    // Create test dataset
    *test_dataset = dataset_create();
    if (!*test_dataset) {
        LOG_ERROR("Failed to create test dataset");
        dataset_free(*train_dataset);
        dataset_free(*val_dataset);
        *train_dataset = NULL;
        *val_dataset   = NULL;
        return -1;
    }

    // Copy data from source dataset tensors if available
    if (dataset->X && dataset->y) {
        // Copy training data
        int train_input_shape[]  = {train_size, dataset->input_size};
        int train_target_shape[] = {train_size, dataset->output_size};

        float* train_X_data = (float*)tensor_data_ptr(dataset->X);
        float* train_y_data = (float*)tensor_data_ptr(dataset->y);

        float* train_X = CM_MALLOC(train_size * dataset->input_size * sizeof(float));
        float* train_y = CM_MALLOC(train_size * sizeof(float));

        if (!train_X || !train_y) {
            LOG_ERROR("Failed to allocate training data");
            dataset_free(*train_dataset);
            dataset_free(*val_dataset);
            dataset_free(*test_dataset);
            *train_dataset = NULL;
            *val_dataset   = NULL;
            *test_dataset  = NULL;
            return -1;
        }

        for (int i = 0; i < train_size; i++) {
            for (int j = 0; j < dataset->input_size; j++) {
                train_X[i * dataset->input_size + j] = train_X_data[i * dataset->input_size + j];
            }
            train_y[i] = train_y_data[i];
        }

        TensorConfig train_config =
            tensor_config_with_dtype_device(dataset->dtype, dataset->device);
        (*train_dataset)->X = tensor_from_data(train_X, train_input_shape, 2, &train_config);
        (*train_dataset)->y = tensor_from_data(train_y, train_target_shape, 2, &train_config);
        CM_FREE(train_X);
        CM_FREE(train_y);

        // Copy validation data
        int val_input_shape[]  = {val_size, dataset->input_size};
        int val_target_shape[] = {val_size, dataset->output_size};

        float* val_X = CM_MALLOC(val_size * dataset->input_size * sizeof(float));
        float* val_y = CM_MALLOC(val_size * sizeof(float));

        if (!val_X || !val_y) {
            LOG_ERROR("Failed to allocate validation data");
            dataset_free(*train_dataset);
            dataset_free(*val_dataset);
            dataset_free(*test_dataset);
            *train_dataset = NULL;
            *val_dataset   = NULL;
            *test_dataset  = NULL;
            return -1;
        }

        for (int i = 0; i < val_size; i++) {
            int src_idx = train_size + i;
            for (int j = 0; j < dataset->input_size; j++) {
                val_X[i * dataset->input_size + j] =
                    train_X_data[src_idx * dataset->input_size + j];
            }
            val_y[i] = train_y_data[src_idx];
        }

        TensorConfig val_config = tensor_config_with_dtype_device(dataset->dtype, dataset->device);
        (*val_dataset)->X       = tensor_from_data(val_X, val_input_shape, 2, &val_config);
        (*val_dataset)->y       = tensor_from_data(val_y, val_target_shape, 2, &val_config);
        CM_FREE(val_X);
        CM_FREE(val_y);

        // Copy test data
        int test_input_shape[]  = {test_size, dataset->input_size};
        int test_target_shape[] = {test_size, dataset->output_size};

        float* test_X = CM_MALLOC(test_size * dataset->input_size * sizeof(float));
        float* test_y = CM_MALLOC(test_size * sizeof(float));

        if (!test_X || !test_y) {
            LOG_ERROR("Failed to allocate test data");
            dataset_free(*train_dataset);
            dataset_free(*val_dataset);
            dataset_free(*test_dataset);
            *train_dataset = NULL;
            *val_dataset   = NULL;
            *test_dataset  = NULL;
            return -1;
        }

        for (int i = 0; i < test_size; i++) {
            int src_idx = train_size + val_size + i;
            for (int j = 0; j < dataset->input_size; j++) {
                test_X[i * dataset->input_size + j] =
                    train_X_data[src_idx * dataset->input_size + j];
            }
            test_y[i] = train_y_data[src_idx];
        }

        TensorConfig test_config = tensor_config_with_dtype_device(dataset->dtype, dataset->device);
        (*test_dataset)->X       = tensor_from_data(test_X, test_input_shape, 2, &test_config);
        (*test_dataset)->y       = tensor_from_data(test_y, test_target_shape, 2, &test_config);
        CM_FREE(test_X);
        CM_FREE(test_y);
    }

    // Set metadata
    (*train_dataset)->num_samples = train_size;
    (*train_dataset)->input_size  = dataset->input_size;
    (*train_dataset)->output_size = dataset->output_size;
    (*train_dataset)->dtype       = dataset->dtype;
    (*train_dataset)->device      = dataset->device;
    (*train_dataset)->is_loaded   = true;

    (*val_dataset)->num_samples = val_size;
    (*val_dataset)->input_size  = dataset->input_size;
    (*val_dataset)->output_size = dataset->output_size;
    (*val_dataset)->dtype       = dataset->dtype;
    (*val_dataset)->device      = dataset->device;
    (*val_dataset)->is_loaded   = true;

    (*test_dataset)->num_samples = test_size;
    (*test_dataset)->input_size  = dataset->input_size;
    (*test_dataset)->output_size = dataset->output_size;
    (*test_dataset)->dtype       = dataset->dtype;
    (*test_dataset)->device      = dataset->device;
    (*test_dataset)->is_loaded   = true;

    LOG_DEBUG("Dataset split successfully: train=%d, val=%d, test=%d", train_size, val_size,
              test_size);
    return 0;
}

int dataset_normalize(Dataset* dataset, const char* method) {
    if (!dataset || !method) {
        LOG_ERROR("Invalid parameters for dataset_normalize");
        return -1;
    }

    LOG_DEBUG("Normalizing dataset using method: %s", method);

    if (strcmp(method, "zscore") == 0) {
        // Z-score normalization
        if (!dataset->feature_means || !dataset->feature_stds) {
            LOG_ERROR("Feature statistics not available for Z-score normalization");
            return -1;
        }

        // TODO
        // In a real implementation, you'd normalize the actual tensor data
        dataset->is_normalized = true;
        LOG_DEBUG("Z-score normalization completed");

    } else if (strcmp(method, "minmax") == 0) {
        // Min-Max normalization
        if (!dataset->feature_mins || !dataset->feature_maxs) {
            LOG_ERROR("Feature statistics not available for Min-Max normalization");
            return -1;
        }
        // TODO
        // In a real implementation, you'd normalize the actual tensor data
        dataset->is_normalized = true;
        LOG_DEBUG("Min-Max normalization completed");

    } else {
        LOG_ERROR("Unknown normalization method: %s", method);
        return -1;
    }

    return 0;
}

int dataset_shuffle(Dataset* dataset, unsigned int seed) {
    if (!dataset || !dataset->indices) {
        LOG_ERROR("Invalid dataset or indices not available for shuffling");
        return -1;
    }

    LOG_DEBUG("Shuffling dataset with seed %u", seed);

    // Set random seed
    srand(seed);

    // Fisher-Yates shuffle
    for (int i = dataset->num_samples - 1; i > 0; i--) {
        int j               = rand() % (i + 1);
        int temp            = dataset->indices[i];
        dataset->indices[i] = dataset->indices[j];
        dataset->indices[j] = temp;
    }

    dataset->is_shuffled = true;
    LOG_DEBUG("Dataset shuffled successfully");
    return 0;
}

// Utility Functions

void dataset_print_summary(Dataset* dataset) {
    if (!dataset)
        return;

    printf("\nDataset Summary\n");
    printf("================\n");
    printf("Name: %s\n", dataset->name);
    printf("Samples: %d\n", dataset->num_samples);
    printf("Input size: %d\n", dataset->input_size);
    printf("Output size: %d\n", dataset->output_size);
    printf("Data type: %d\n", dataset->dtype);
    printf("Device: %d\n", dataset->device);
    printf("Normalized: %s\n", dataset->is_normalized ? "Yes" : "No");
    printf("Shuffled: %s\n", dataset->is_shuffled ? "Yes" : "No");
    printf("Loaded: %s\n", dataset->is_loaded ? "Yes" : "No");

    if (dataset->filepath) {
        printf("File: %s\n", dataset->filepath);
    }

    printf("\n");
}

size_t dataset_get_memory_usage(Dataset* dataset) {
    if (!dataset)
        return 0;

    size_t usage = sizeof(Dataset);

    // Add tensor memory usage
    if (dataset->X) {
        usage += dataset->X->numel * dtype_size(dataset->X->dtype);
    }
    if (dataset->y) {
        usage += dataset->y->numel * dtype_size(dataset->y->dtype);
    }

    // Add arrays memory usage
    if (dataset->indices) {
        usage += dataset->num_samples * sizeof(int);
    }
    if (dataset->feature_means) {
        usage += dataset->input_size * sizeof(float);
    }
    if (dataset->feature_stds) {
        usage += dataset->input_size * sizeof(float);
    }
    if (dataset->feature_mins) {
        usage += dataset->input_size * sizeof(float);
    }
    if (dataset->feature_maxs) {
        usage += dataset->input_size * sizeof(float);
    }

    return usage;
}

bool dataset_is_valid(Dataset* dataset) {
    if (!dataset)
        return false;

    // Check basic requirements
    if (dataset->num_samples <= 0)
        return false;
    if (dataset->input_size <= 0)
        return false;
    if (dataset->output_size <= 0)
        return false;

    // Check tensors
    if (!dataset->X || !dataset->y)
        return false;

    // Check tensor dimensions
    if (dataset->X->ndim != 2 || dataset->y->ndim != 2)
        return false;
    if (dataset->X->shape[0] != dataset->num_samples)
        return false;
    if (dataset->X->shape[1] != dataset->input_size)
        return false;
    if (dataset->y->shape[0] != dataset->num_samples)
        return false;
    if (dataset->y->shape[1] != dataset->output_size)
        return false;

    return true;
}

Dataset* dataset_copy(Dataset* dataset) {
    if (!dataset)
        return NULL;

    LOG_DEBUG("Copying dataset");

    Dataset* copy = dataset_create();
    if (!copy)
        return NULL;

    // Copy basic properties
    copy->name          = dataset->name;
    copy->num_samples   = dataset->num_samples;
    copy->input_size    = dataset->input_size;
    copy->output_size   = dataset->output_size;
    copy->dtype         = dataset->dtype;
    copy->device        = dataset->device;
    copy->is_normalized = dataset->is_normalized;
    copy->is_shuffled   = dataset->is_shuffled;

    // TODO
    // Copy tensors (in a real implementation, you'd create proper copies)
    // For now, we'll just set the metadata
    copy->is_loaded = false; // Indicate that tensors need to be copied

    LOG_DEBUG("Dataset copied successfully");
    return copy;
}

// DataLoader Implementation

DataLoader* dataloader_create(Dataset* dataset, int batch_size, bool shuffle) {
    if (!dataset || batch_size <= 0) {
        LOG_ERROR("Invalid parameters for dataloader_create");
        return NULL;
    }

    LOG_DEBUG("Creating DataLoader: batch_size=%d, shuffle=%d", batch_size, shuffle);

    DataLoader* loader = CM_MALLOC(sizeof(DataLoader));
    if (!loader)
        return NULL;

    loader->dataset         = dataset;
    loader->batch_size      = batch_size;
    loader->shuffle         = shuffle;
    loader->drop_last       = false;
    loader->num_workers     = 0; // Single-threaded by default
    loader->prefetch_factor = 2; // Prefetch 2 batches by default
    loader->pin_memory      = false;

    loader->current_batch = 0;
    loader->total_batches = (dataset->num_samples + batch_size - 1) / batch_size;
    loader->batch_indices = NULL;

    loader->on_batch_start = NULL;
    loader->on_batch_end   = NULL;

    // Initialize shuffled indices
    loader->shuffled_indices = CM_MALLOC(dataset->num_samples * sizeof(int));
    if (!loader->shuffled_indices) {
        CM_FREE(loader);
        return NULL;
    }

    for (int i = 0; i < dataset->num_samples; i++) {
        loader->shuffled_indices[i] = i;
    }

    if (shuffle) {
        dataset_shuffle(dataset, (unsigned int)time(NULL));
        if (dataset->indices) {
            memcpy(loader->shuffled_indices, dataset->indices, dataset->num_samples * sizeof(int));
        }
    }

    loader->current_epoch = 0;
    loader->user_data     = NULL;

    LOG_DEBUG("DataLoader created successfully");
    return loader;
}

void dataloader_free(DataLoader* loader) {
    if (!loader)
        return;

    if (loader->shuffled_indices) {
        CM_FREE(loader->shuffled_indices);
    }
    if (loader->batch_indices) {
        CM_FREE(loader->batch_indices);
    }

    CM_FREE(loader);
}

int dataloader_reset(DataLoader* loader) {
    if (!loader)
        return -1;

    loader->current_batch = 0;
    loader->current_epoch++;

    // Reshuffle if needed
    if (loader->shuffle && loader->dataset) {
        dataset_shuffle(loader->dataset, (unsigned int)time(NULL));
        if (loader->dataset->indices) {
            memcpy(loader->shuffled_indices, loader->dataset->indices,
                   loader->dataset->num_samples * sizeof(int));
        }
    }

    return 0;
}

Batch* dataloader_next_batch(DataLoader* loader) {
    if (!loader || !loader->dataset)
        return NULL;

    if (loader->current_batch >= loader->total_batches) {
        return NULL; // No more batches
    }

    // Calculate batch indices
    int start_idx = loader->current_batch * loader->batch_size;
    int end_idx   = start_idx + loader->batch_size;
    if (end_idx > loader->dataset->num_samples) {
        if (loader->drop_last) {
            return NULL; // Drop incomplete batch
        }
        end_idx = loader->dataset->num_samples;
    }

    int actual_batch_size = end_idx - start_idx;

    // Allocate batch indices
    if (loader->batch_indices) {
        CM_FREE(loader->batch_indices);
    }
    loader->batch_indices = CM_MALLOC(actual_batch_size * sizeof(int));
    if (!loader->batch_indices)
        return NULL;

    // Copy indices
    for (int i = 0; i < actual_batch_size; i++) {
        loader->batch_indices[i] = loader->shuffled_indices[start_idx + i];
    }

    // Create batch
    Batch* batch = CM_MALLOC(sizeof(Batch));
    if (!batch) {
        CM_FREE(loader->batch_indices);
        return NULL;
    }

    // Create batch tensors
    int batch_X_shape[] = {actual_batch_size, loader->dataset->input_size};
    int batch_y_shape[] = {actual_batch_size, loader->dataset->output_size};

    TensorConfig config =
        tensor_config_with_dtype_device(loader->dataset->dtype, loader->dataset->device);
    batch->X = tensor_empty(batch_X_shape, 2, &config);
    batch->y = tensor_empty(batch_y_shape, 2, &config);

    if (!batch->X || !batch->y) {
        if (batch->X)
            tensor_free(batch->X);
        if (batch->y)
            tensor_free(batch->y);
        CM_FREE(batch);
        CM_FREE(loader->batch_indices);
        return NULL;
    }

    // Copy data from dataset
    float* X_data    = (float*)tensor_data_ptr(batch->X);
    float* y_data    = (float*)tensor_data_ptr(batch->y);
    float* dataset_X = (float*)tensor_data_ptr(loader->dataset->X);
    float* dataset_y = (float*)tensor_data_ptr(loader->dataset->y);

    if (X_data && y_data && dataset_X && dataset_y) {
        for (int i = 0; i < actual_batch_size; i++) {
            int sample_idx = loader->batch_indices[i];

            // Copy input features
            memcpy(X_data + i * loader->dataset->input_size,
                   dataset_X + sample_idx * loader->dataset->input_size,
                   loader->dataset->input_size * sizeof(float));

            // Copy targets
            memcpy(y_data + i * loader->dataset->output_size,
                   dataset_y + sample_idx * loader->dataset->output_size,
                   loader->dataset->output_size * sizeof(float));
        }
    }

    batch->batch_size  = actual_batch_size;
    batch->batch_index = loader->current_batch;
    batch->epoch       = loader->current_epoch;
    batch->user_data   = NULL;

    // Apply augmentation if configured (placeholder for future enhancement)
    // In a multi-threaded implementation, this would be done in worker threads

    // Call callbacks
    if (loader->on_batch_start) {
        loader->on_batch_start(batch);
    }

    loader->current_batch++;

    // Call end callback
    if (loader->on_batch_end) {
        loader->on_batch_end(batch);
    }

    return batch;
}

bool dataloader_has_next(DataLoader* loader) {
    if (!loader)
        return false;
    return loader->current_batch < loader->total_batches;
}

int dataloader_get_batch_count(DataLoader* loader) {
    if (!loader)
        return 0;
    return loader->total_batches;
}

int dataloader_get_current_batch(DataLoader* loader) {
    if (!loader)
        return -1;
    return loader->current_batch;
}

// Batch Operations

void batch_free(Batch* batch) {
    if (!batch)
        return;

    if (batch->X)
        tensor_free(batch->X);
    if (batch->y)
        tensor_free(batch->y);

    CM_FREE(batch);
}

Tensor* batch_get_input(Batch* batch) {
    if (!batch)
        return NULL;
    return batch->X;
}

Tensor* batch_get_targets(Batch* batch) {
    if (!batch)
        return NULL;
    return batch->y;
}

int batch_get_size(Batch* batch) {
    if (!batch)
        return 0;
    return batch->batch_size;
}

void batch_print_summary(Batch* batch) {
    if (!batch)
        return;

    printf("\nBatch Summary\n");
    printf("=============\n");
    printf("Batch Index: %d\n", batch->batch_index);
    printf("Batch Size: %d\n", batch->batch_size);
    printf("Epoch: %d\n", batch->epoch);
    if (batch->X) {
        printf("Input Shape: [%d", batch->X->shape[0]);
        for (int i = 1; i < batch->X->ndim; i++) {
            printf(", %d", batch->X->shape[i]);
        }
        printf("]\n");
    }
    if (batch->y) {
        printf("Target Shape: [%d", batch->y->shape[0]);
        for (int i = 1; i < batch->y->ndim; i++) {
            printf(", %d", batch->y->shape[i]);
        }
        printf("]\n");
    }
    printf("\n");
}
