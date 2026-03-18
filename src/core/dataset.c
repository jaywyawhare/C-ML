#define _POSIX_C_SOURCE 200809L
#include "core/dataset.h"
#include "datasets/datasets.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include "core/config.h"
#include "backend/threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

typedef struct PrefetchQueue {
    Batch** batches;
    int* batch_indices;
    int capacity;
    int size;
    int front;
    int rear;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    bool shutdown;
} PrefetchQueue;

typedef struct WorkerContext {
    DataLoader* loader;
    PrefetchQueue* queue;
    int worker_id;
    ThreadPool* thread_pool;
} WorkerContext;

static PrefetchQueue* prefetch_queue_create(int capacity) {
    PrefetchQueue* queue = malloc(sizeof(PrefetchQueue));
    if (!queue)
        return NULL;

    queue->batches       = calloc((size_t)capacity, sizeof(Batch*));
    queue->batch_indices = calloc((size_t)capacity, sizeof(int));
    if (!queue->batches || !queue->batch_indices) {
        if (queue->batches)
            free(queue->batches);
        if (queue->batch_indices)
            free(queue->batch_indices);
        free(queue);
        return NULL;
    }

    queue->capacity = capacity;
    queue->size     = 0;
    queue->front    = 0;
    queue->rear     = 0;
    queue->shutdown = false;

    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);

    return queue;
}

static void prefetch_queue_free(PrefetchQueue* queue) {
    if (!queue)
        return;

    pthread_mutex_lock(&queue->mutex);
    queue->shutdown = true;
    pthread_cond_broadcast(&queue->not_empty);
    pthread_cond_broadcast(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);

    while (queue->size > 0) {
        Batch* batch = queue->batches[queue->front];
        if (batch) {
            if (batch->X)
                tensor_free(batch->X);
            if (batch->y)
                tensor_free(batch->y);
            free(batch);
        }
        queue->front = (queue->front + 1) % queue->capacity;
        queue->size--;
    }

    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);

    if (queue->batches)
        free(queue->batches);
    if (queue->batch_indices)
        free(queue->batch_indices);
    free(queue);
}

static int prefetch_queue_enqueue(PrefetchQueue* queue, Batch* batch, int batch_idx) {
    if (!queue || !batch)
        return -1;

    pthread_mutex_lock(&queue->mutex);

    while (queue->size >= queue->capacity && !queue->shutdown) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }

    if (queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }

    queue->batches[queue->rear]       = batch;
    queue->batch_indices[queue->rear] = batch_idx;
    queue->rear                       = (queue->rear + 1) % queue->capacity;
    queue->size++;

    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);

    return 0;
}

static Batch* prefetch_queue_dequeue(PrefetchQueue* queue, int* batch_idx) {
    if (!queue)
        return NULL;

    pthread_mutex_lock(&queue->mutex);

    while (queue->size == 0 && !queue->shutdown) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }

    if (queue->shutdown && queue->size == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return NULL;
    }

    Batch* batch = queue->batches[queue->front];
    if (batch_idx) {
        *batch_idx = queue->batch_indices[queue->front];
    }
    queue->front = (queue->front + 1) % queue->capacity;
    queue->size--;

    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);

    return batch;
}

static Batch* load_batch_at_index(DataLoader* loader, int batch_idx) {
    if (!loader || !loader->dataset || batch_idx >= loader->total_batches) {
        return NULL;
    }

    int start_idx = batch_idx * loader->batch_size;
    int end_idx   = start_idx + loader->batch_size;
    if (end_idx > loader->dataset->num_samples) {
        if (loader->drop_last) {
            return NULL;
        }
        end_idx = loader->dataset->num_samples;
    }

    int actual_batch_size = end_idx - start_idx;

    Batch* batch = malloc(sizeof(Batch));
    if (!batch)
        return NULL;
    int batch_X_shape[] = {actual_batch_size, loader->dataset->input_size};
    int batch_y_shape[] = {actual_batch_size, loader->dataset->output_size};

    TensorConfig config = {.dtype      = loader->dataset->dtype,
                           .device     = loader->dataset->device,
                           .has_dtype  = true,
                           .has_device = true};
    batch->X            = tensor_empty(batch_X_shape, 2, &config);
    batch->y            = tensor_empty(batch_y_shape, 2, &config);

    if (!batch->X || !batch->y) {
        if (batch->X)
            tensor_free(batch->X);
        if (batch->y)
            tensor_free(batch->y);
        free(batch);
        return NULL;
    }

    float* X_data    = (float*)tensor_data_ptr(batch->X);
    float* y_data    = (float*)tensor_data_ptr(batch->y);
    float* dataset_X = (float*)tensor_data_ptr(loader->dataset->X);
    float* dataset_y = (float*)tensor_data_ptr(loader->dataset->y);

    if (X_data && y_data && dataset_X && dataset_y) {
        for (int i = 0; i < actual_batch_size; i++) {
            int sample_idx = loader->shuffled_indices[start_idx + i];
            memcpy(X_data + i * loader->dataset->input_size,
                   dataset_X + sample_idx * loader->dataset->input_size,
                   (size_t)loader->dataset->input_size * sizeof(float));
            memcpy(y_data + i * loader->dataset->output_size,
                   dataset_y + sample_idx * loader->dataset->output_size,
                   (size_t)loader->dataset->output_size * sizeof(float));
        }
    }
    batch->batch_size  = actual_batch_size;
    batch->batch_index = batch_idx;
    batch->epoch       = loader->current_epoch;
    batch->user_data   = NULL;

    return batch;
}
static void* worker_prefetch_batches(void* arg) {
    WorkerContext* ctx   = (WorkerContext*)arg;
    DataLoader* loader   = ctx->loader;
    PrefetchQueue* queue = ctx->queue;

    int prefetch_count = loader->prefetch_factor;
    int next_batch     = 0;

    while (!queue->shutdown) {
        for (int i = 0; i < prefetch_count; i++) {
            int batch_idx = next_batch++;
            if (batch_idx >= loader->total_batches) {
                break; // No more batches
            }

            Batch* batch = load_batch_at_index(loader, batch_idx);
            if (batch) {
                prefetch_queue_enqueue(queue, batch, batch_idx);
            }
        }
        if (next_batch >= loader->total_batches) {
            break;
        }
        struct timespec ts = {0, 1000000}; // 1ms
        nanosleep(&ts, NULL);
    }

    return NULL;
}

Dataset* dataset_create(void) {
    Dataset* dataset = malloc(sizeof(Dataset));
    if (!dataset) {
        LOG_ERROR("Failed to allocate memory for Dataset");
        return NULL;
    }

    dataset->name        = "Dataset";
    dataset->X           = NULL;
    dataset->y           = NULL;
    dataset->num_samples = 0;
    dataset->input_size  = 0;
    dataset->output_size = 0;

    dataset->dtype         = DTYPE_FLOAT32;
    dataset->device        = DEVICE_CPU;
    dataset->is_normalized = false;
    dataset->is_shuffled   = false;

    dataset->feature_means = NULL;
    dataset->feature_stds  = NULL;
    dataset->feature_mins  = NULL;
    dataset->feature_maxs  = NULL;

    dataset->feature_names = NULL;
    dataset->class_names   = NULL;
    dataset->num_classes   = 0;

    dataset->indices   = NULL;
    dataset->is_loaded = false;
    dataset->filepath  = NULL;
    dataset->user_data = NULL;

    return dataset;
}

Dataset* dataset_from_arrays(float* X, float* y, int num_samples, int input_size, int output_size) {
    Dataset* dataset = dataset_create();
    if (!dataset)
        return NULL;

    if (dataset_load_arrays(dataset, X, y, num_samples, input_size, output_size) != 0) {
        LOG_ERROR("Failed to load arrays into dataset");
        dataset_free(dataset);
        return NULL;
    }

    extern void cml_track_dataset(Dataset*);
    cml_track_dataset(dataset);

    return dataset;
}

int dataset_load_arrays(Dataset* dataset, float* X, float* y, int num_samples, int input_size,
                        int output_size) {
    if (!dataset || !X || !y) {
        LOG_ERROR("Invalid parameters for dataset_load_arrays");
        return -1;
    }

    int X_shape[2]      = {num_samples, input_size};
    TensorConfig config = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    dataset->X = tensor_from_data(X, X_shape, 2, &config);
    if (!dataset->X) {
        LOG_ERROR("Failed to create input tensor");
        return -1;
    }

    int y_shape[2] = {num_samples, output_size};
    dataset->y     = tensor_from_data(y, y_shape, 2, &config);
    if (!dataset->y) {
        LOG_ERROR("Failed to create target tensor");
        tensor_free(dataset->X);
        dataset->X = NULL;
        return -1;
    }

    dataset->num_samples = num_samples;
    dataset->input_size  = input_size;
    dataset->output_size = output_size;
    dataset->is_loaded   = true;

    dataset->indices = malloc((size_t)num_samples * sizeof(int));
    if (!dataset->indices) {
        LOG_WARNING("Failed to allocate indices array, shuffling will be disabled");
    } else {
        for (int i = 0; i < num_samples; i++) {
            dataset->indices[i] = i;
        }
    }

    return 0;
}

int dataset_load_file(Dataset* dataset, const char* filepath, const char* format) {
    if (!dataset || !filepath || !format) {
        LOG_ERROR("Invalid parameters for dataset_load_file");
        return -1;
    }

    if (strcmp(format, "csv") == 0 || strcmp(format, "CSV") == 0) {
        float* X = NULL;
        float* y = NULL;
        int num_samples = 0, num_features = 0, num_classes = 0;
        char** class_names = NULL;

        if (cml_csv_parse(filepath, -1, &X, &y, &num_samples, &num_features,
                          &num_classes, &class_names) != 0) {
            LOG_ERROR("Failed to parse CSV file: %s", filepath);
            return -1;
        }

        int result = dataset_load_arrays(dataset, X, y, num_samples, num_features, 1);
        free(X);
        free(y);

        if (result != 0) {
            if (class_names) {
                for (int i = 0; i < num_classes; i++)
                    free(class_names[i]);
                free(class_names);
            }
            return -1;
        }

        dataset->num_classes = num_classes;
        dataset->class_names = class_names;
        dataset->filepath = strdup(filepath);

        cml_dataset_compute_stats(dataset);

        LOG_INFO("CSV file loaded: %d samples, %d features, %d classes",
                  num_samples, num_features, num_classes);
        return 0;
    }

    LOG_ERROR("Unsupported file format: %s (supported: csv)", format);
    return -1;
}

int dataset_get_statistics(Dataset* dataset, void* stats) {
    if (!dataset) {
        LOG_ERROR("Invalid dataset for dataset_get_statistics");
        return -1;
    }

    if (!dataset->X || dataset->num_samples == 0 || dataset->input_size == 0) {
        LOG_ERROR("Dataset has no data for statistics computation");
        return -1;
    }

    /* Compute statistics if not already present */
    if (!dataset->feature_means || !dataset->feature_stds ||
        !dataset->feature_mins || !dataset->feature_maxs) {
        cml_dataset_compute_stats(dataset);
    }

    /* If caller provided a stats buffer, copy the four arrays into it.
     * Layout: float[4][input_size] = { means, stds, mins, maxs } */
    if (stats && dataset->feature_means && dataset->feature_stds &&
        dataset->feature_mins && dataset->feature_maxs) {
        size_t sz = (size_t)dataset->input_size * sizeof(float);
        float* out = (float*)stats;
        memcpy(out, dataset->feature_means, sz);
        memcpy(out + dataset->input_size, dataset->feature_stds, sz);
        memcpy(out + 2 * dataset->input_size, dataset->feature_mins, sz);
        memcpy(out + 3 * dataset->input_size, dataset->feature_maxs, sz);
    }

    return 0;
}

void dataset_free(Dataset* dataset) {
    if (!dataset)
        return;

    if (dataset->X)
        tensor_free(dataset->X);
    if (dataset->y)
        tensor_free(dataset->y);

    if (dataset->feature_means)
        free(dataset->feature_means);
    if (dataset->feature_stds)
        free(dataset->feature_stds);
    if (dataset->feature_mins)
        free(dataset->feature_mins);
    if (dataset->feature_maxs)
        free(dataset->feature_maxs);

    if (dataset->feature_names) {
        for (int i = 0; i < dataset->input_size; i++) {
            if (dataset->feature_names[i])
                free(dataset->feature_names[i]);
        }
        free(dataset->feature_names);
    }

    if (dataset->class_names) {
        for (int i = 0; i < dataset->num_classes; i++) {
            if (dataset->class_names[i])
                free(dataset->class_names[i]);
        }
        free(dataset->class_names);
    }

    if (dataset->indices)
        free(dataset->indices);
    if (dataset->filepath)
        free(dataset->filepath);

    free(dataset);
}

int dataset_split(Dataset* dataset, float train_ratio, Dataset** train_dataset,
                  Dataset** val_dataset) {
    if (!dataset || !train_dataset || !val_dataset) {
        LOG_ERROR("Invalid parameters for dataset_split");
        return -1;
    }

    if (train_ratio <= 0.0f || train_ratio >= 1.0f) {
        LOG_ERROR("Invalid train_ratio: %.2f (must be between 0 and 1)", (double)train_ratio);
        return -1;
    }

    int train_size = (int)((float)dataset->num_samples * train_ratio);
    int val_size   = dataset->num_samples - train_size;

    *train_dataset = dataset_create();
    if (!*train_dataset) {
        LOG_ERROR("Failed to create training dataset");
        return -1;
    }

    *val_dataset = dataset_create();
    if (!*val_dataset) {
        LOG_ERROR("Failed to create validation dataset");
        dataset_free(*train_dataset);
        *train_dataset = NULL;
        return -1;
    }
    (*train_dataset)->num_samples   = train_size;
    (*train_dataset)->input_size    = dataset->input_size;
    (*train_dataset)->output_size   = dataset->output_size;
    (*train_dataset)->dtype         = dataset->dtype;
    (*train_dataset)->device        = dataset->device;
    (*train_dataset)->num_classes   = dataset->num_classes;
    (*train_dataset)->is_normalized = dataset->is_normalized;
    (*train_dataset)->class_names   = dataset->class_names;

    (*val_dataset)->num_samples   = val_size;
    (*val_dataset)->input_size    = dataset->input_size;
    (*val_dataset)->output_size   = dataset->output_size;
    (*val_dataset)->dtype         = dataset->dtype;
    (*val_dataset)->device        = dataset->device;
    (*val_dataset)->num_classes   = dataset->num_classes;
    (*val_dataset)->is_normalized = dataset->is_normalized;
    (*val_dataset)->class_names   = dataset->class_names;

    /* Copy normalization statistics if present */
    if (dataset->feature_mins && dataset->feature_maxs) {
        size_t stat_sz                 = (size_t)dataset->input_size * sizeof(float);
        (*train_dataset)->feature_mins = malloc(stat_sz);
        (*train_dataset)->feature_maxs = malloc(stat_sz);
        (*val_dataset)->feature_mins   = malloc(stat_sz);
        (*val_dataset)->feature_maxs   = malloc(stat_sz);
        if ((*train_dataset)->feature_mins && (*train_dataset)->feature_maxs) {
            memcpy((*train_dataset)->feature_mins, dataset->feature_mins, stat_sz);
            memcpy((*train_dataset)->feature_maxs, dataset->feature_maxs, stat_sz);
        }
        if ((*val_dataset)->feature_mins && (*val_dataset)->feature_maxs) {
            memcpy((*val_dataset)->feature_mins, dataset->feature_mins, stat_sz);
            memcpy((*val_dataset)->feature_maxs, dataset->feature_maxs, stat_sz);
        }
    }
    if (dataset->feature_means && dataset->feature_stds) {
        size_t stat_sz                  = (size_t)dataset->input_size * sizeof(float);
        (*train_dataset)->feature_means = malloc(stat_sz);
        (*train_dataset)->feature_stds  = malloc(stat_sz);
        (*val_dataset)->feature_means   = malloc(stat_sz);
        (*val_dataset)->feature_stds    = malloc(stat_sz);
        if ((*train_dataset)->feature_means && (*train_dataset)->feature_stds) {
            memcpy((*train_dataset)->feature_means, dataset->feature_means, stat_sz);
            memcpy((*train_dataset)->feature_stds, dataset->feature_stds, stat_sz);
        }
        if ((*val_dataset)->feature_means && (*val_dataset)->feature_stds) {
            memcpy((*val_dataset)->feature_means, dataset->feature_means, stat_sz);
            memcpy((*val_dataset)->feature_stds, dataset->feature_stds, stat_sz);
        }
    }

    if (dataset->X && dataset->y) {
        int train_input_shape[]  = {train_size, dataset->input_size};
        int train_target_shape[] = {train_size, dataset->output_size};

        float* source_X_data = (float*)tensor_data_ptr(dataset->X);
        float* source_y_data = (float*)tensor_data_ptr(dataset->y);

        if (!source_X_data || !source_y_data) {
            LOG_ERROR("Failed to access source tensor data");
            dataset_free(*train_dataset);
            dataset_free(*val_dataset);
            *train_dataset = NULL;
            *val_dataset   = NULL;
            return -1;
        }
        float* train_X = malloc((size_t)train_size * (size_t)dataset->input_size * sizeof(float));
        float* train_y = malloc((size_t)train_size * (size_t)dataset->output_size * sizeof(float));

        if (!train_X || !train_y) {
            LOG_ERROR("Failed to allocate training data");
            if (train_X)
                free(train_X);
            if (train_y)
                free(train_y);
            dataset_free(*train_dataset);
            dataset_free(*val_dataset);
            *train_dataset = NULL;
            *val_dataset   = NULL;
            return -1;
        }

        for (int i = 0; i < train_size; i++) {
            memcpy(train_X + i * dataset->input_size, source_X_data + i * dataset->input_size,
                   (size_t)dataset->input_size * sizeof(float));
            memcpy(train_y + i * dataset->output_size, source_y_data + i * dataset->output_size,
                   (size_t)dataset->output_size * sizeof(float));
        }

        TensorConfig train_config = {.dtype      = dataset->dtype,
                                     .device     = dataset->device,
                                     .has_dtype  = true,
                                     .has_device = true};
        (*train_dataset)->X       = tensor_from_data(train_X, train_input_shape, 2, &train_config);
        (*train_dataset)->y       = tensor_from_data(train_y, train_target_shape, 2, &train_config);
        free(train_X);
        free(train_y);

        if (!(*train_dataset)->X || !(*train_dataset)->y) {
            LOG_ERROR("Failed to create training tensors");
            dataset_free(*train_dataset);
            dataset_free(*val_dataset);
            *train_dataset = NULL;
            *val_dataset   = NULL;
            return -1;
        }
        float* val_X = malloc((size_t)val_size * (size_t)dataset->input_size * sizeof(float));
        float* val_y = malloc((size_t)val_size * (size_t)dataset->output_size * sizeof(float));

        if (!val_X || !val_y) {
            LOG_ERROR("Failed to allocate validation data");
            if (val_X)
                free(val_X);
            if (val_y)
                free(val_y);
            dataset_free(*train_dataset);
            dataset_free(*val_dataset);
            *train_dataset = NULL;
            *val_dataset   = NULL;
            return -1;
        }

        for (int i = 0; i < val_size; i++) {
            int src_idx = train_size + i;
            memcpy(val_X + i * dataset->input_size, source_X_data + src_idx * dataset->input_size,
                   (size_t)dataset->input_size * sizeof(float));
            memcpy(val_y + i * dataset->output_size, source_y_data + src_idx * dataset->output_size,
                   (size_t)dataset->output_size * sizeof(float));
        }

        int val_input_shape[]   = {val_size, dataset->input_size};
        int val_target_shape[]  = {val_size, dataset->output_size};
        TensorConfig val_config = {.dtype      = dataset->dtype,
                                   .device     = dataset->device,
                                   .has_dtype  = true,
                                   .has_device = true};
        (*val_dataset)->X       = tensor_from_data(val_X, val_input_shape, 2, &val_config);
        (*val_dataset)->y       = tensor_from_data(val_y, val_target_shape, 2, &val_config);
        free(val_X);
        free(val_y);

        if (!(*val_dataset)->X || !(*val_dataset)->y) {
            LOG_ERROR("Failed to create validation tensors");
            dataset_free(*train_dataset);
            dataset_free(*val_dataset);
            *train_dataset = NULL;
            *val_dataset   = NULL;
            return -1;
        }

        (*train_dataset)->is_loaded = true;
        (*val_dataset)->is_loaded   = true;
    } else {
        (*train_dataset)->is_loaded = false;
        (*val_dataset)->is_loaded   = false;
    }

    return 0;
}

int dataset_split_three(Dataset* dataset, float train_ratio, float val_ratio,
                        Dataset** train_dataset, Dataset** val_dataset, Dataset** test_dataset) {
    if (!dataset || !train_dataset || !val_dataset || !test_dataset) {
        LOG_ERROR("Invalid parameters for dataset_split_three");
        return -1;
    }

    if (train_ratio <= 0.0f || train_ratio >= 1.0f) {
        LOG_ERROR("Invalid train_ratio: %.2f (must be between 0 and 1)", (double)train_ratio);
        return -1;
    }

    if (val_ratio <= 0.0f || val_ratio >= 1.0f) {
        LOG_ERROR("Invalid val_ratio: %.2f (must be between 0 and 1)", (double)val_ratio);
        return -1;
    }

    if (train_ratio + val_ratio >= 1.0f) {
        LOG_ERROR("Invalid ratios: train_ratio + val_ratio (%.2f) must be < 1.0",
                  (double)(train_ratio + val_ratio));
        return -1;
    }


    int train_size = (int)((float)dataset->num_samples * train_ratio);
    int val_size   = (int)((float)dataset->num_samples * val_ratio);
    int test_size  = dataset->num_samples - train_size - val_size;

    *train_dataset = dataset_create();
    if (!*train_dataset) {
        LOG_ERROR("Failed to create training dataset");
        return -1;
    }

    *val_dataset = dataset_create();
    if (!*val_dataset) {
        LOG_ERROR("Failed to create validation dataset");
        dataset_free(*train_dataset);
        *train_dataset = NULL;
        return -1;
    }

    *test_dataset = dataset_create();
    if (!*test_dataset) {
        LOG_ERROR("Failed to create test dataset");
        dataset_free(*train_dataset);
        dataset_free(*val_dataset);
        *train_dataset = NULL;
        *val_dataset   = NULL;
        return -1;
    }

    if (dataset->X && dataset->y) {
        int train_input_shape[]  = {train_size, dataset->input_size};
        int train_target_shape[] = {train_size, dataset->output_size};

        float* train_X_data = (float*)tensor_data_ptr(dataset->X);
        float* train_y_data = (float*)tensor_data_ptr(dataset->y);

        float* train_X = malloc((size_t)train_size * (size_t)dataset->input_size * sizeof(float));
        float* train_y = malloc((size_t)train_size * sizeof(float));

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

        TensorConfig train_config = {.dtype      = dataset->dtype,
                                     .device     = dataset->device,
                                     .has_dtype  = true,
                                     .has_device = true};
        (*train_dataset)->X       = tensor_from_data(train_X, train_input_shape, 2, &train_config);
        (*train_dataset)->y       = tensor_from_data(train_y, train_target_shape, 2, &train_config);
        free(train_X);
        free(train_y);
        int val_input_shape[]  = {val_size, dataset->input_size};
        int val_target_shape[] = {val_size, dataset->output_size};

        float* val_X = malloc((size_t)val_size * (size_t)dataset->input_size * sizeof(float));
        float* val_y = malloc((size_t)val_size * sizeof(float));

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

        TensorConfig val_config = {.dtype      = dataset->dtype,
                                   .device     = dataset->device,
                                   .has_dtype  = true,
                                   .has_device = true};
        (*val_dataset)->X       = tensor_from_data(val_X, val_input_shape, 2, &val_config);
        (*val_dataset)->y       = tensor_from_data(val_y, val_target_shape, 2, &val_config);
        free(val_X);
        free(val_y);
        int test_input_shape[]  = {test_size, dataset->input_size};
        int test_target_shape[] = {test_size, dataset->output_size};

        float* test_X = malloc((size_t)test_size * (size_t)dataset->input_size * sizeof(float));
        float* test_y = malloc((size_t)test_size * sizeof(float));

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

        TensorConfig test_config = {.dtype      = dataset->dtype,
                                    .device     = dataset->device,
                                    .has_dtype  = true,
                                    .has_device = true};
        (*test_dataset)->X       = tensor_from_data(test_X, test_input_shape, 2, &test_config);
        (*test_dataset)->y       = tensor_from_data(test_y, test_target_shape, 2, &test_config);
        free(test_X);
        free(test_y);
    }
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


    return 0;
}

int dataset_normalize(Dataset* dataset, const char* method) {
    if (!dataset || !method) {
        LOG_ERROR("Invalid parameters for dataset_normalize");
        return -1;
    }


    if (strcmp(method, "zscore") == 0) {
        // Z-score normalization: (x - mean) / std
        if (!dataset->feature_means || !dataset->feature_stds) {
            LOG_ERROR("Feature statistics not available for Z-score normalization");
            return -1;
        }

        if (!dataset->X) {
            LOG_ERROR("Dataset tensor X is not available");
            return -1;
        }

        float* X_data = (float*)tensor_data_ptr(dataset->X);
        if (!X_data) {
            LOG_ERROR("Failed to access tensor data");
            return -1;
        }
        for (int i = 0; i < dataset->num_samples; i++) {
            for (int j = 0; j < dataset->input_size; j++) {
                float mean = dataset->feature_means[j];
                float std  = dataset->feature_stds[j];
                if (std > 1e-8f) { // Avoid division by zero
                    int idx     = i * dataset->input_size + j;
                    X_data[idx] = (X_data[idx] - mean) / std;
                }
            }
        }

        dataset->is_normalized = true;


    } else if (strcmp(method, "minmax") == 0) {
        // Min-Max normalization: (x - min) / (max - min)
        if (!dataset->feature_mins || !dataset->feature_maxs) {
            LOG_ERROR("Feature statistics not available for Min-Max normalization");
            return -1;
        }

        if (!dataset->X) {
            LOG_ERROR("Dataset tensor X is not available");
            return -1;
        }

        float* X_data = (float*)tensor_data_ptr(dataset->X);
        if (!X_data) {
            LOG_ERROR("Failed to access tensor data");
            return -1;
        }
        for (int i = 0; i < dataset->num_samples; i++) {
            for (int j = 0; j < dataset->input_size; j++) {
                float min_val = dataset->feature_mins[j];
                float max_val = dataset->feature_maxs[j];
                float range   = max_val - min_val;
                if (range > 1e-8f) { // Avoid division by zero
                    int idx     = i * dataset->input_size + j;
                    X_data[idx] = (X_data[idx] - min_val) / range;
                }
            }
        }

        dataset->is_normalized = true;


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


    srand(seed);
    for (int i = dataset->num_samples - 1; i > 0; i--) {
        int j               = rand() % (i + 1);
        int temp            = dataset->indices[i];
        dataset->indices[i] = dataset->indices[j];
        dataset->indices[j] = temp;
    }

    dataset->is_shuffled = true;
    return 0;
}

void dataset_print_summary(Dataset* dataset) {
    if (!dataset)
        return;

    printf("\nDataset Summary\n");
    printf("\n");
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

    if (dataset->X) {
        usage += dataset->X->numel * cml_dtype_size(dataset->X->dtype);
    }
    if (dataset->y) {
        usage += dataset->y->numel * cml_dtype_size(dataset->y->dtype);
    }
    if (dataset->indices) {
        usage += (size_t)dataset->num_samples * sizeof(int);
    }
    if (dataset->feature_means) {
        usage += (size_t)dataset->input_size * sizeof(float);
    }
    if (dataset->feature_stds) {
        usage += (size_t)dataset->input_size * sizeof(float);
    }
    if (dataset->feature_mins) {
        usage += (size_t)dataset->input_size * sizeof(float);
    }
    if (dataset->feature_maxs) {
        usage += (size_t)dataset->input_size * sizeof(float);
    }

    return usage;
}

bool dataset_is_valid(Dataset* dataset) {
    if (!dataset)
        return false;
    if (dataset->num_samples <= 0)
        return false;
    if (dataset->input_size <= 0)
        return false;
    if (dataset->output_size <= 0)
        return false;

    if (!dataset->X || !dataset->y)
        return false;

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


    Dataset* copy = dataset_create();
    if (!copy)
        return NULL;

    copy->name          = dataset->name;
    copy->num_samples   = dataset->num_samples;
    copy->input_size    = dataset->input_size;
    copy->output_size   = dataset->output_size;
    copy->dtype         = dataset->dtype;
    copy->device        = dataset->device;
    copy->is_normalized = dataset->is_normalized;
    copy->is_shuffled   = dataset->is_shuffled;

    if (dataset->X && dataset->y) {
        copy->X = tensor_clone(dataset->X);
        copy->y = tensor_clone(dataset->y);

        if (!copy->X || !copy->y) {
            LOG_ERROR("Failed to clone dataset tensors");
            if (copy->X)
                tensor_free(copy->X);
            if (copy->y)
                tensor_free(copy->y);
            dataset_free(copy);
            return NULL;
        }

        copy->is_loaded = true;
    } else {
        copy->X         = NULL;
        copy->y         = NULL;
        copy->is_loaded = false;
    }

    if (dataset->indices) {
        copy->indices = malloc((size_t)dataset->num_samples * sizeof(int));
        if (copy->indices) {
            memcpy(copy->indices, dataset->indices, (size_t)dataset->num_samples * sizeof(int));
        }
    }

    return copy;
}

DataLoader* dataloader_create(Dataset* dataset, int batch_size, bool shuffle) {
    if (!dataset || batch_size <= 0) {
        LOG_ERROR("Invalid parameters for dataloader_create");
        return NULL;
    }


    DataLoader* loader = malloc(sizeof(DataLoader));
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

    loader->shuffled_indices = malloc((size_t)dataset->num_samples * sizeof(int));
    if (!loader->shuffled_indices) {
        free(loader);
        return NULL;
    }

    for (int i = 0; i < dataset->num_samples; i++) {
        loader->shuffled_indices[i] = i;
    }

    if (shuffle) {
        dataset_shuffle(dataset, (unsigned int)time(NULL));
        if (dataset->indices) {
            memcpy(loader->shuffled_indices, dataset->indices,
                   (size_t)dataset->num_samples * sizeof(int));
        }
    }

    loader->current_epoch = 0;
    loader->user_data     = NULL;

    return loader;
}

void dataloader_free(DataLoader* loader) {
    if (!loader)
        return;
    if (loader->num_workers > 0 && loader->prefetch_queue) {
        PrefetchQueue* queue = (PrefetchQueue*)loader->prefetch_queue;
        prefetch_queue_free(queue);
        if (loader->worker_threads) {
            pthread_t* threads = (pthread_t*)loader->worker_threads;
            for (int i = 0; i < loader->num_active_workers; i++) {
                pthread_join(threads[i], NULL);
            }
            free(threads);
        }

        if (loader->worker_contexts) {
            free(loader->worker_contexts);
        }
    }

    if (loader->shuffled_indices) {
        free(loader->shuffled_indices);
    }
    if (loader->batch_indices) {
        free(loader->batch_indices);
    }

    free(loader);
}

int dataloader_reset(DataLoader* loader) {
    if (!loader)
        return -1;

    loader->current_batch = 0;
    loader->current_epoch++;
    if (loader->shuffle && loader->dataset) {
        dataset_shuffle(loader->dataset, (unsigned int)time(NULL));
        if (loader->dataset->indices) {
            memcpy(loader->shuffled_indices, loader->dataset->indices,
                   (size_t)loader->dataset->num_samples * sizeof(int));
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
    if (loader->num_workers > 0 && loader->prefetch_queue) {
        PrefetchQueue* queue = (PrefetchQueue*)loader->prefetch_queue;
        int batch_idx;
        Batch* batch = prefetch_queue_dequeue(queue, &batch_idx);

        if (batch) {
            loader->current_batch = batch_idx + 1;
            return batch;
        }
    }
    int start_idx = loader->current_batch * loader->batch_size;
    int end_idx   = start_idx + loader->batch_size;
    if (end_idx > loader->dataset->num_samples) {
        if (loader->drop_last) {
            return NULL; // Drop incomplete batch
        }
        end_idx = loader->dataset->num_samples;
    }

    Batch* batch = load_batch_at_index(loader, loader->current_batch);
    if (batch) {
        if (loader->on_batch_start) {
            loader->on_batch_start(batch);
        }

        loader->current_batch++;
        if (loader->on_batch_end) {
            loader->on_batch_end(batch);
        }
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

void batch_free(Batch* batch) {
    if (!batch)
        return;

    if (batch->X)
        tensor_free(batch->X);
    if (batch->y)
        tensor_free(batch->y);

    free(batch);
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
    printf("\n");
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

DataLoader* dataloader_create_with_workers(Dataset* dataset, int batch_size, bool shuffle,
                                           int num_workers) {
    if (!dataset || batch_size <= 0) {
        LOG_ERROR("Invalid parameters for dataloader_create_with_workers");
        return NULL;
    }

    DataLoader* loader = dataloader_create(dataset, batch_size, shuffle);
    if (!loader) {
        return NULL;
    }

    loader->prefetch_queue     = NULL;
    loader->worker_threads     = NULL;
    loader->worker_contexts    = NULL;
    loader->num_active_workers = 0;

    if (num_workers > 0) {
        loader->num_workers = num_workers;
        int queue_capacity   = loader->prefetch_factor * num_workers;
        PrefetchQueue* queue = prefetch_queue_create(queue_capacity);
        if (!queue) {
            LOG_WARNING("Failed to create prefetch queue, falling back to single-threaded");
            loader->num_workers = 0;
            return loader;
        }

        loader->prefetch_queue = queue;
        pthread_t* threads      = calloc((size_t)num_workers, sizeof(pthread_t));
        WorkerContext* contexts = calloc((size_t)num_workers, sizeof(WorkerContext));

        if (!threads || !contexts) {
            if (threads)
                free(threads);
            if (contexts)
                free(contexts);
            prefetch_queue_free(queue);
            loader->num_workers    = 0;
            loader->prefetch_queue = NULL;
            LOG_WARNING("Failed to allocate worker threads, falling back to single-threaded");
            return loader;
        }

        loader->worker_threads  = threads;
        loader->worker_contexts = contexts;
        for (int i = 0; i < num_workers; i++) {
            contexts[i].loader      = loader;
            contexts[i].queue       = queue;
            contexts[i].worker_id   = i;
            contexts[i].thread_pool = NULL;

            if (pthread_create(&threads[i], NULL, worker_prefetch_batches, &contexts[i]) != 0) {
                LOG_WARNING("Failed to create worker thread %d", i);
            } else {
                loader->num_active_workers++;
            }
        }
    } else {
        loader->num_workers = 0;
        /* single-threaded fallback */
    }

    return loader;
}

int dataloader_get_batch_tensors(DataLoader* loader, Tensor*** batch_inputs,
                                 Tensor*** batch_targets) {
    if (!loader || !batch_inputs || !batch_targets) {
        return 0;
    }

    Batch* batch = dataloader_next_batch(loader);
    if (!batch) {
        return 0;
    }
    *batch_inputs  = malloc(sizeof(Tensor*));
    *batch_targets = malloc(sizeof(Tensor*));
    if (!*batch_inputs || !*batch_targets) {
        batch_free(batch);
        return 0;
    }

    (*batch_inputs)[0]  = batch_get_input(batch);
    (*batch_targets)[0] = batch_get_targets(batch);

    int batch_size = batch_get_size(batch);
    batch_free(batch);
    return batch_size;
}

int dataloader_for_each(DataLoader* loader, BatchCallback callback, void* user_data) {
    if (!loader || !callback) {
        return -1;
    }

    dataloader_reset(loader);

    Tensor** batch_inputs  = NULL;
    Tensor** batch_targets = NULL;
    int batch_size         = 0;

    while ((batch_size = dataloader_get_batch_tensors(loader, &batch_inputs, &batch_targets)) > 0) {
        if (callback(batch_inputs, batch_targets, batch_size, user_data) != 0) {
            if (batch_inputs && batch_inputs[0]) {
                tensor_free(batch_inputs[0]);
            }
            if (batch_targets && batch_targets[0]) {
                tensor_free(batch_targets[0]);
            }
            free(batch_inputs);
            free(batch_targets);
            return -1;
        }
        if (batch_inputs && batch_inputs[0]) {
            tensor_free(batch_inputs[0]);
        }
        if (batch_targets && batch_targets[0]) {
            tensor_free(batch_targets[0]);
        }
        free(batch_inputs);
        free(batch_targets);
    }

    return 0;
}

Dataset* dataset_xor(void) {
    float X[4][2] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
    float y[4]    = {0.0f, 1.0f, 1.0f, 0.0f};
    float* X_flat = malloc(4 * 2 * sizeof(float));
    float* y_flat = malloc(4 * sizeof(float));
    if (!X_flat || !y_flat) {
        free(X_flat);
        free(y_flat);
        return NULL;
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            X_flat[i * 2 + j] = X[i][j];
        }
        y_flat[i] = y[i];
    }

    Dataset* dataset = dataset_from_arrays(X_flat, y_flat, 4, 2, 1);
    if (dataset) {
        dataset->name = "XOR";
    }

    free(X_flat);
    free(y_flat);
    return dataset;
}

Dataset* dataset_random_classification(int num_samples, int num_features, int num_classes) {
    if (num_samples <= 0 || num_features <= 0 || num_classes <= 0) {
        return NULL;
    }
    cml_random_seed();

    float* X = malloc((size_t)num_samples * (size_t)num_features * sizeof(float));
    float* y = malloc((size_t)num_samples * sizeof(float));
    if (!X || !y) {
        free(X);
        free(y);
        return NULL;
    }
    for (int i = 0; i < num_samples * num_features; i++) {
        X[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < num_samples; i++) {
        y[i] = (float)(rand() % num_classes);
    }

    Dataset* dataset = dataset_from_arrays(X, y, num_samples, num_features, 1);
    if (dataset) {
        dataset->name        = "RandomClassification";
        dataset->num_classes = num_classes;
    }

    free(X);
    free(y);
    return dataset;
}

int transform_normalize(Dataset* dataset, float* mean, float* std) {
    if (!dataset || !dataset->X) {
        LOG_ERROR("Invalid dataset for transform_normalize");
        return -1;
    }
    if (!mean || !std) {
        return dataset_normalize(dataset, "zscore");
    }
    float* X_data = (float*)tensor_data_ptr(dataset->X);
    if (!X_data) {
        LOG_ERROR("Failed to access tensor data");
        return -1;
    }
    for (int i = 0; i < dataset->num_samples; i++) {
        for (int j = 0; j < dataset->input_size; j++) {
            float m = mean[j];
            float s = std[j];
            if (s < 1e-8f) { // Avoid division by zero
                s = 1.0f;
            }
            int idx     = i * dataset->input_size + j;
            X_data[idx] = (X_data[idx] - m) / s;
        }
    }
    if (!dataset->feature_means) {
        dataset->feature_means = malloc((size_t)dataset->input_size * sizeof(float));
    }
    if (!dataset->feature_stds) {
        dataset->feature_stds = malloc((size_t)dataset->input_size * sizeof(float));
    }

    if (dataset->feature_means && dataset->feature_stds) {
        memcpy(dataset->feature_means, mean, (size_t)dataset->input_size * sizeof(float));
        memcpy(dataset->feature_stds, std, (size_t)dataset->input_size * sizeof(float));
    }

    dataset->is_normalized = true;


    return 0;
}
