/**
 * @file datasets.h
 * @brief Dataset Hub - Keras/PyTorch-style dataset loading
 *
 * One-liner dataset loading with automatic download and caching:
 *
 *   Dataset* ds = cml_dataset_load("iris");
 *   Dataset* ds = cml_dataset_load("mnist");
 *   Dataset* ds = cml_dataset_from_csv("data.csv", -1);
 */

#ifndef CML_DATASETS_H
#define CML_DATASETS_H

#include "core/dataset.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load a named dataset (downloads + caches automatically)
 *
 * Supported datasets:
 *   "iris"           - 150 samples, 4 features, 3 classes
 *   "wine"           - 178 samples, 13 features, 3 classes
 *   "breast_cancer"  - 569 samples, 30 features, 2 classes
 *   "boston"          - 506 samples, 13 features, regression
 *   "mnist"          - 70000 samples, 784 features, 10 classes (IDX format)
 *   "fashion_mnist"  - 70000 samples, 784 features, 10 classes (IDX format)
 *   "cifar10"        - 60000 samples, 3072 features, 10 classes
 *   "airline"        - 144 samples, 1 feature, time series (built-in)
 *   "digits"         - 1797 samples, 64 features, 10 classes (built-in)
 *
 * @param name Dataset name (case-insensitive)
 * @return Dataset*, or NULL on failure
 */
Dataset* cml_dataset_load(const char* name);

/**
 * @brief Load dataset from a CSV file
 *
 * Auto-detects: header row, delimiter (comma/semicolon), string labels.
 *
 * @param filepath Path to CSV file
 * @param target_col Target column index (-1 = last, 0 = first, etc.)
 * @return Dataset*, or NULL on failure
 */
Dataset* cml_dataset_from_csv(const char* filepath, int target_col);

/**
 * @brief Download a file to the cache directory
 *
 * Uses curl or wget. Returns cached path if file already exists.
 *
 * @param url URL to download
 * @param filename Local filename in cache dir
 * @return Cached file path (static buffer), or NULL on failure
 */
const char* cml_dataset_download(const char* url, const char* filename);

/**
 * @brief Get the dataset cache directory (default: ~/.cml/datasets/)
 */
const char* cml_dataset_cache_dir(void);

/**
 * @brief Set the dataset cache directory
 */
void cml_dataset_set_cache_dir(const char* dir);

/** Compute feature statistics (min/max/mean/std) for normalization */
void cml_dataset_compute_stats(Dataset* ds);

/** Load IDX-3 ubyte images -> float array normalized to [0,1] */
float* cml_idx_load_images(const char* path, int* n, int* rows, int* cols);

/** Load IDX-1 ubyte labels -> float array of class indices */
float* cml_idx_load_labels(const char* path, int* n);

/** Parse CSV file into float arrays + optional string-label mapping */
int cml_csv_parse(const char* filepath, int target_col,
                  float** X_out, float** y_out,
                  int* num_samples, int* num_features, int* num_classes,
                  char*** class_names_out);

/** Airline passengers (144 monthly values, 1949-1960) */
const float* cml_builtin_airline_data(int* n);

/** Digits 8x8 dataset (1797 samples, 64 features) */
const float* cml_builtin_digits_data(int* n, int* features);
const float* cml_builtin_digits_labels(int* n);

#ifdef __cplusplus
}
#endif

#endif /* CML_DATASETS_H */
