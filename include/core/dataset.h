#ifndef CML_CORE_DATASET_H
#define CML_CORE_DATASET_H

#include "tensor/tensor.h"
#include "core/logging.h"
#include "alloc/memory_management.h"
#include "core/augmentation.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Dataset;
struct DataLoader;
struct Batch;

typedef struct Dataset {
    const char* name;

    Tensor* X;
    Tensor* y;
    int num_samples;
    int input_size;
    int output_size;

    DType dtype;
    DeviceType device;
    bool is_normalized;
    bool is_shuffled;

    float* feature_means;
    float* feature_stds;
    float* feature_mins;
    float* feature_maxs;

    char** feature_names;
    char** class_names;
    int num_classes;

    int* indices;
    bool is_loaded;
    char* filepath;
    void* user_data;
} Dataset;

typedef struct DataLoader {
    Dataset* dataset;

    int batch_size;      // Batch size
    bool shuffle;        // Whether to shuffle data
    bool drop_last;      // Whether to drop incomplete batches
    int num_workers;     // Number of worker threads
    int prefetch_factor; // Number of batches to prefetch
    bool pin_memory;     // Whether to pin memory (for GPU)

    // Iteration state
    int current_batch;  // Current batch index
    int total_batches;  // Total number of batches
    int* batch_indices; // Indices for current batch

    // Callbacks
    void (*on_batch_start)(struct Batch* batch); // Called at start of each batch
    void (*on_batch_end)(struct Batch* batch);   // Called at end of each batch

    // Internal state
    int* shuffled_indices; // Shuffled sample indices
    int current_epoch;     // Current epoch
    void* user_data;       // User-defined data

    // Multi-worker support (internal)
    void* prefetch_queue;   // PrefetchQueue* (opaque pointer)
    void* worker_threads;   // pthread_t* array for workers
    void* worker_contexts;  // WorkerContext* array
    int num_active_workers; // Number of active worker threads
} DataLoader;

typedef struct Batch {
    Tensor* X;       // Batch input features
    Tensor* y;       // Batch targets
    int batch_size;  // Actual batch size
    int batch_index; // Batch index
    int epoch;       // Current epoch
    void* user_data; // User-defined data
} Batch;

Dataset* dataset_create(void);
Dataset* dataset_from_arrays(float* X, float* y, int num_samples, int input_size, int output_size);
int dataset_load_arrays(Dataset* dataset, float* X, float* y, int num_samples, int input_size,
                        int output_size);
int dataset_load_file(Dataset* dataset, const char* filepath, const char* format);
void dataset_free(Dataset* dataset);

int dataset_split(Dataset* dataset, float train_ratio, Dataset** train_dataset,
                  Dataset** val_dataset);
int dataset_split_three(Dataset* dataset, float train_ratio, float val_ratio,
                        Dataset** train_dataset, Dataset** val_dataset, Dataset** test_dataset);
int dataset_normalize(Dataset* dataset, const char* method);
int dataset_shuffle(Dataset* dataset, unsigned int seed);
int dataset_get_statistics(Dataset* dataset, void* stats);

DataLoader* dataloader_create(Dataset* dataset, int batch_size, bool shuffle);
void dataloader_free(DataLoader* dataloader);
int dataloader_reset(DataLoader* dataloader);

Batch* dataloader_next_batch(DataLoader* dataloader);
bool dataloader_has_next(DataLoader* dataloader);
int dataloader_get_batch_count(DataLoader* dataloader);
int dataloader_get_current_batch(DataLoader* dataloader);

void batch_free(Batch* batch);
Tensor* batch_get_input(Batch* batch);
Tensor* batch_get_targets(Batch* batch);
int batch_get_size(Batch* batch);

void dataset_print_summary(Dataset* dataset);
void batch_print_summary(Batch* batch);
size_t dataset_get_memory_usage(Dataset* dataset);
bool dataset_is_valid(Dataset* dataset);
Dataset* dataset_copy(Dataset* dataset);

DataLoader* dataloader_create_with_workers(Dataset* dataset, int batch_size, bool shuffle,
                                           int num_workers);
int dataloader_get_batch_tensors(DataLoader* loader, Tensor*** batch_inputs,
                                 Tensor*** batch_targets);

typedef int (*BatchCallback)(Tensor** inputs, Tensor** targets, int batch_size, void* user_data);
int dataloader_for_each(DataLoader* loader, BatchCallback callback, void* user_data);

Dataset* dataset_xor(void);
Dataset* dataset_random_classification(int num_samples, int num_features, int num_classes);

/* mean/std NULL for auto-compute */
int transform_normalize(Dataset* dataset, float* mean, float* std);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_DATASET_H
