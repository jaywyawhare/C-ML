# Dataset Hub

C-ML provides a dataset hub for one-liner dataset loading with automatic download and caching.

## Quick Start

```c
#include "cml.h"

int main(void) {
    cml_init();

    Dataset* ds = cml_dataset_load("iris");
    dataset_normalize(ds, "minmax");

    Dataset *train, *test;
    dataset_split(ds, 0.8f, &train, &test);

    Tensor* X = train->X;   // [num_samples, input_size]
    Tensor* y = train->y;   // [num_samples, output_size]

    printf("Samples: %d, Features: %d, Classes: %d\n",
           ds->num_samples, ds->input_size, ds->num_classes);

    dataset_free(train);
    dataset_free(test);
    dataset_free(ds);
    cml_cleanup();
    return 0;
}
```

## Supported Datasets

| Name | Samples | Features | Classes | Type | Source |
|------|---------|----------|---------|------|--------|
| `iris` | 150 | 4 | 3 | Classification | UCI (download) |
| `wine` | 178 | 13 | 3 | Classification | UCI (download) |
| `breast_cancer` | 569 | 30 | 2 | Classification | UCI (download) |
| `boston` | 506 | 13 | -- | Regression | UCI (download) |
| `mnist` | 70,000 | 784 | 10 | Classification | IDX format (download) |
| `fashion_mnist` | 70,000 | 784 | 10 | Classification | IDX format (download) |
| `cifar10` | 60,000 | 3,072 | 10 | Classification | Binary (download) |
| `airline` | 144 | 1 | -- | Time series | Built-in |
| `digits` | 1,797 | 64 | 10 | Classification | Built-in |

Dataset names are case-insensitive.

## API Reference

### Loading

```c
Dataset* cml_dataset_load(const char* name);

// target_col: -1 = last column, 0 = first, etc.
Dataset* cml_dataset_from_csv(const char* filepath, int target_col);
```

CSV loading auto-detects: header rows, delimiters (comma/semicolon), and string labels (encoded as integers).

### Preprocessing

```c
void dataset_normalize(Dataset* ds, const char* method);
void dataset_split(Dataset* ds, float ratio, Dataset** train, Dataset** test);
void cml_dataset_compute_stats(Dataset* ds);
```

### Cache Management

```c
const char* cml_dataset_cache_dir(void);
void cml_dataset_set_cache_dir(const char* dir);
```

Datasets are downloaded on first use via `curl` or `wget` and cached locally. Subsequent loads use the cached copy.

### Memory

```c
void dataset_free(Dataset* ds);
```

## Dataset Structure

```c
typedef struct Dataset {
    Tensor* X;           // Input features [num_samples, input_size]
    Tensor* y;           // Target values  [num_samples, output_size]
    int num_samples;
    int input_size;
    int output_size;
    int num_classes;     // 0 for regression
    bool is_normalized;
    // ... feature statistics
} Dataset;
```

## Examples

### Classification (Iris)

```c
Dataset* ds = cml_dataset_load("iris");
dataset_normalize(ds, "minmax");
Dataset *train, *test;
dataset_split(ds, 0.8f, &train, &test);
// train->X: [120, 4], train->y: [120, 1]
// 3 classes (setosa, versicolor, virginica)
```

### Regression (Boston Housing)

```c
Dataset* ds = cml_dataset_load("boston");
dataset_normalize(ds, "zscore");
Dataset *train, *test;
dataset_split(ds, 0.8f, &train, &test);
```

### Time Series (Airline)

```c
Dataset* ds = cml_dataset_load("airline");
```

### Custom CSV

```c
Dataset* ds = cml_dataset_from_csv("my_data.csv", -1);
Dataset* ds = cml_dataset_from_csv("my_data.csv", 0);
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `CML_DATASET_DIR` | `~/.cml/datasets/` | Cache directory for downloaded datasets |
