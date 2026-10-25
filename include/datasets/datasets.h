#ifndef CML_DATASETS_H
#define CML_DATASETS_H

#include "core/dataset.h"

#ifdef __cplusplus
extern "C" {
#endif

Dataset* cml_dataset_load(const char* name);

/* How to read `/`- or `.`-separated dates. ISO `YYYY-MM-DD` is always accepted. */
typedef enum {
    CML_CSV_DATE_ISO = 0, /* only ISO 8601 dates are auto-parsed */
    CML_CSV_DATE_DMY,     /* day/month/year */
    CML_CSV_DATE_MDY,     /* month/day/year */
} CSVDateOrder;

/* Options for cml_csv_parse_ex (see cml_csv_default_options for the defaults). */
typedef struct {
    int target_col;          /* label column; negative counts from the end */
    int header;              /* -1 auto-detect, 0 none, 1 present */
    char delimiter;          /* 0 to auto-detect, else the field separator */
    CSVDateOrder date_order; /* reading of non-ISO dates */
    int expand_datetime;     /* split date/time columns into numeric components */
    int cyclical;            /* also emit sin/cos pairs for periodic components */
} CSVParseOptions;

/* Auto-detects header row, delimiter, and string labels; expands date/time columns. */
Dataset* cml_dataset_from_csv(const char* filepath, int target_col);

/* Returns cached path (static buffer) if file already exists. */
const char* cml_dataset_download(const char* url, const char* filename);

const char* cml_dataset_cache_dir(void);

void cml_dataset_set_cache_dir(const char* dir);

void cml_dataset_compute_stats(Dataset* ds);

float* cml_idx_load_images(const char* path, int* n, int* rows, int* cols);

float* cml_idx_load_labels(const char* path, int* n);

int cml_csv_parse(const char* filepath, int target_col, float** X_out, float** y_out,
                  int* num_samples, int* num_features, int* num_classes, char*** class_names_out);

/* Default parse options: last column target, auto header/delimiter, ISO dates,
 * date/time expansion on, no cyclical encoding. */
CSVParseOptions cml_csv_default_options(void);

/* Like cml_csv_parse with full control. When feature_names_out is non-NULL it
 * receives num_features heap-allocated feature names (caller frees each, then the
 * array). All output arrays are cml_malloc'd. Returns 0 on success. */
int cml_csv_parse_ex(const char* filepath, const CSVParseOptions* opts, float** X_out,
                     float** y_out, int* num_samples, int* num_features, int* num_classes,
                     char*** class_names_out, char*** feature_names_out);

const float* cml_builtin_airline_data(int* n);

const float* cml_builtin_digits_data(int* n, int* features);
const float* cml_builtin_digits_labels(int* n);

#ifdef __cplusplus
}
#endif

#endif /* CML_DATASETS_H */
