#ifndef CML_DATASETS_H
#define CML_DATASETS_H

#include "core/dataset.h"

#ifdef __cplusplus
extern "C" {
#endif

Dataset* cml_dataset_load(const char* name);

/* Auto-detects header row, delimiter, and string labels. */
Dataset* cml_dataset_from_csv(const char* filepath, int target_col);

/* Returns cached path (static buffer) if file already exists. */
const char* cml_dataset_download(const char* url, const char* filename);

const char* cml_dataset_cache_dir(void);

void cml_dataset_set_cache_dir(const char* dir);

void cml_dataset_compute_stats(Dataset* ds);

float* cml_idx_load_images(const char* path, int* n, int* rows, int* cols);

float* cml_idx_load_labels(const char* path, int* n);

int cml_csv_parse(const char* filepath, int target_col,
                  float** X_out, float** y_out,
                  int* num_samples, int* num_features, int* num_classes,
                  char*** class_names_out);

const float* cml_builtin_airline_data(int* n);

const float* cml_builtin_digits_data(int* n, int* features);
const float* cml_builtin_digits_labels(int* n);

#ifdef __cplusplus
}
#endif

#endif /* CML_DATASETS_H */
