/**
 * @file dataset.h
 * @brief High-level dataset API for data loading
 *
 * This header provides a dataset interface that abstracts
 * away the complexity of data management and provides easy data loading.
 */

#ifndef CML_CORE_DATASET_H
#define CML_CORE_DATASET_H

#include "tensor/tensor.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include "Core/augmentation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct Dataset;
struct DataLoader;
struct Batch;

/**
 * @brief Dataset structure for managing training data
 *
 * This structure encapsulates input data, target data, and metadata
 * needed for training neural networks.
 */
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

/**
 * @brief DataLoader for batch processing
 *
 * This structure handles batching, shuffling, and iteration over datasets
 */
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
} DataLoader;

// Batch Structure

/**
 * @brief Batch structure for batch processing
 *
 * Contains a batch of data with input features and targets
 */
typedef struct Batch {
    Tensor* X;       // Batch input features
    Tensor* y;       // Batch targets
    int batch_size;  // Actual batch size
    int batch_index; // Batch index
    int epoch;       // Current epoch
    void* user_data; // User-defined data
} Batch;

// Dataset Creation and Management

/**
 * @brief Create a new dataset
 *
 * @return New dataset, or NULL on failure
 */
Dataset* dataset_create(void);

/**
 * @brief Create dataset from arrays
 *
 * @param X Input features array
 * @param y Target labels array
 * @param num_samples Number of samples
 * @param input_size Input feature size
 * @param output_size Output feature size
 * @return New dataset, or NULL on failure
 */
Dataset* dataset_from_arrays(float* X, float* y, int num_samples, int input_size, int output_size);

/**
 * @brief Load dataset from arrays into existing dataset
 *
 * @param dataset Target dataset
 * @param X Input features array
 * @param y Target labels array
 * @param num_samples Number of samples
 * @param input_size Input feature size
 * @param output_size Output feature size
 * @return 0 on success, negative value on failure
 */
int dataset_load_arrays(Dataset* dataset, float* X, float* y, int num_samples, int input_size,
                        int output_size);

/**
 * @brief Load dataset from file
 *
 * @param dataset Target dataset
 * @param filepath Path to data file
 * @param format File format (CSV, NPY, etc.)
 * @return 0 on success, negative value on failure
 */
int dataset_load_file(Dataset* dataset, const char* filepath, const char* format);

/**
 * @brief Free dataset and all resources
 *
 * @param dataset Dataset to free
 */
void dataset_free(Dataset* dataset);

// Dataset Operations

/**
 * @brief Split dataset into training and validation sets
 *
 * @param dataset Source dataset
 * @param train_ratio Ratio of training data (0.0 to 1.0)
 * @param train_dataset Training dataset output
 * @param val_dataset Validation dataset output
 * @return 0 on success, negative value on failure
 */
int dataset_split(Dataset* dataset, float train_ratio, Dataset** train_dataset,
                  Dataset** val_dataset);

/**
 * @brief Split dataset into training, validation, and test sets
 *
 * @param dataset Source dataset
 * @param train_ratio Ratio of training data (0.0 to 1.0)
 * @param val_ratio Ratio of validation data (0.0 to 1.0)
 * @param train_dataset Training dataset output
 * @param val_dataset Validation dataset output
 * @param test_dataset Test dataset output
 * @return 0 on success, negative value on failure
 */
int dataset_split_three(Dataset* dataset, float train_ratio, float val_ratio,
                        Dataset** train_dataset, Dataset** val_dataset, Dataset** test_dataset);

/**
 * @brief Normalize dataset features
 *
 * @param dataset Target dataset
 * @param method Normalization method (Z-score, Min-Max, etc.)
 * @return 0 on success, negative value on failure
 */
int dataset_normalize(Dataset* dataset, const char* method);

/**
 * @brief Shuffle dataset
 *
 * @param dataset Target dataset
 * @param seed Random seed for shuffling
 * @return 0 on success, negative value on failure
 */
int dataset_shuffle(Dataset* dataset, unsigned int seed);

/**
 * @brief Get dataset statistics
 *
 * @param dataset Target dataset
 * @param stats Statistics structure to fill (reserved for future use)
 * @return 0 on success, negative value on failure
 */
int dataset_get_statistics(Dataset* dataset, void* stats);

// DataLoader Creation and Management

/**
 * @brief Create a new DataLoader
 *
 * @param dataset Source dataset
 * @param batch_size Batch size
 * @param shuffle Whether to shuffle data
 * @return New DataLoader, or NULL on failure
 */
DataLoader* dataloader_create(Dataset* dataset, int batch_size, bool shuffle);

/**
 * @brief Free DataLoader and all resources
 *
 * @param dataloader DataLoader to free
 */
void dataloader_free(DataLoader* dataloader);

/**
 * @brief Reset DataLoader for new epoch
 *
 * @param dataloader Target DataLoader
 * @return 0 on success, negative value on failure
 */
int dataloader_reset(DataLoader* dataloader);

// DataLoader Iteration

/**
 * @brief Get next batch from DataLoader
 *
 * @param dataloader Target DataLoader
 * @return Next batch, or NULL if no more batches
 */
Batch* dataloader_next_batch(DataLoader* dataloader);

/**
 * @brief Check if DataLoader has more batches
 *
 * @param dataloader Target DataLoader
 * @return true if more batches available, false otherwise
 */
bool dataloader_has_next(DataLoader* dataloader);

/**
 * @brief Get total number of batches
 *
 * @param dataloader Target DataLoader
 * @return Total number of batches
 */
int dataloader_get_batch_count(DataLoader* dataloader);

/**
 * @brief Get current batch index
 *
 * @param dataloader Target DataLoader
 * @return Current batch index
 */
int dataloader_get_current_batch(DataLoader* dataloader);

// Batch Operations

/**
 * @brief Free batch and all resources
 *
 * @param batch Batch to free
 */
void batch_free(Batch* batch);

/**
 * @brief Get batch input features
 *
 * @param batch Target batch
 * @return Input features tensor
 */
Tensor* batch_get_input(Batch* batch);

/**
 * @brief Get batch targets
 *
 * @param batch Target batch
 * @return Targets tensor
 */
Tensor* batch_get_targets(Batch* batch);

/**
 * @brief Get batch size
 *
 * @param batch Target batch
 * @return Batch size
 */
int batch_get_size(Batch* batch);

// Utility Functions

/**
 * @brief Print dataset summary
 *
 * @param dataset Target dataset
 */
void dataset_print_summary(Dataset* dataset);

/**
 * @brief Print batch summary
 *
 * @param batch Target batch
 */
void batch_print_summary(Batch* batch);

/**
 * @brief Get dataset memory usage
 *
 * @param dataset Target dataset
 * @return Memory usage in bytes
 */
size_t dataset_get_memory_usage(Dataset* dataset);

/**
 * @brief Check if dataset is valid
 *
 * @param dataset Target dataset
 * @return true if valid, false otherwise
 */
bool dataset_is_valid(Dataset* dataset);

/**
 * @brief Copy dataset
 *
 * @param dataset Source dataset
 * @return New dataset copy, or NULL on failure
 */
Dataset* dataset_copy(Dataset* dataset);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_DATASET_H
