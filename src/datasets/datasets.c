/**
 * @file datasets.c
 * @brief Dataset Hub - registry, download, cache, and loading
 */

#define _POSIX_C_SOURCE 200809L
#include "datasets/datasets.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <ctype.h>
#include <math.h>

void cml_dataset_compute_stats(Dataset* ds) {
    if (!ds || !ds->X || ds->num_samples == 0 || ds->input_size == 0) return;

    int n = ds->num_samples;
    int nf = ds->input_size;
    float* X = (float*)tensor_data_ptr(ds->X);
    if (!X) return;

    ds->feature_means = calloc(nf, sizeof(float));
    ds->feature_stds  = calloc(nf, sizeof(float));
    ds->feature_mins  = malloc(sizeof(float) * nf);
    ds->feature_maxs  = malloc(sizeof(float) * nf);
    if (!ds->feature_means || !ds->feature_stds || !ds->feature_mins || !ds->feature_maxs)
        return;

    /* Compute min, max, mean */
    for (int j = 0; j < nf; j++) {
        ds->feature_mins[j] = X[j];
        ds->feature_maxs[j] = X[j];
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < nf; j++) {
            float v = X[i * nf + j];
            ds->feature_means[j] += v;
            if (v < ds->feature_mins[j]) ds->feature_mins[j] = v;
            if (v > ds->feature_maxs[j]) ds->feature_maxs[j] = v;
        }
    }
    for (int j = 0; j < nf; j++)
        ds->feature_means[j] /= n;

    /* Compute std */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < nf; j++) {
            float d = X[i * nf + j] - ds->feature_means[j];
            ds->feature_stds[j] += d * d;
        }
    }
    for (int j = 0; j < nf; j++) {
        ds->feature_stds[j] = sqrtf(ds->feature_stds[j] / n);
    }
}

static char g_cache_dir[512] = "";

const char* cml_dataset_cache_dir(void) {
    if (g_cache_dir[0] == '\0') {
        const char* home = getenv("HOME");
        if (!home) home = "/tmp";
        snprintf(g_cache_dir, sizeof(g_cache_dir), "%s/.cml/datasets", home);
    }
    return g_cache_dir;
}

void cml_dataset_set_cache_dir(const char* dir) {
    if (dir)
        snprintf(g_cache_dir, sizeof(g_cache_dir), "%s", dir);
}

static int ensure_dir(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) return 0;
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", path);
    return system(cmd);
}

const char* cml_dataset_download(const char* url, const char* filename) {
    const char* dir = cml_dataset_cache_dir();
    ensure_dir(dir);

    static char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, filename);

    /* Check cache */
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 0) {
        LOG_INFO("[datasets] Cached: %s", path);
        return path;
    }

    /* Download */
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
             "curl -fsSL -o '%s' '%s' 2>/dev/null || wget -q -O '%s' '%s' 2>/dev/null",
             path, url, path, url);

    LOG_INFO("[datasets] Downloading %s ...", url);
    int ret = system(cmd);
    if (ret != 0) {
        LOG_ERROR("[datasets] Download failed: %s", url);
        /* Remove partial file */
        remove(path);
        return NULL;
    }

    /* Verify non-empty */
    if (stat(path, &st) != 0 || st.st_size == 0) {
        LOG_ERROR("[datasets] Downloaded file is empty: %s", path);
        remove(path);
        return NULL;
    }

    LOG_INFO("[datasets] Saved: %s (%ld bytes)", path, (long)st.st_size);
    return path;
}

/* Helper: decompress .gz file */
static const char* download_and_gunzip(const char* url, const char* gz_name, const char* final_name) {
    const char* dir = cml_dataset_cache_dir();
    static char final_path[1024];
    snprintf(final_path, sizeof(final_path), "%s/%s", dir, final_name);

    /* Check if already decompressed */
    struct stat st;
    if (stat(final_path, &st) == 0 && st.st_size > 0)
        return final_path;

    /* Download .gz */
    const char* gz_path = cml_dataset_download(url, gz_name);
    if (!gz_path) return NULL;

    /* Decompress */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "gunzip -kf '%s' 2>/dev/null || gzip -dkf '%s' 2>/dev/null", gz_path, gz_path);
    system(cmd);

    if (stat(final_path, &st) == 0 && st.st_size > 0)
        return final_path;

    LOG_ERROR("[datasets] Failed to decompress %s", gz_name);
    return NULL;
}

static Dataset* load_iris(void) {
    const char* path = cml_dataset_download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "iris.data");
    if (!path) return NULL;

    float* X = NULL; float* y = NULL;
    int n = 0, nf = 0, nc = 0;
    char** class_names = NULL;
    if (cml_csv_parse(path, -1, &X, &y, &n, &nf, &nc, &class_names) != 0)
        return NULL;

    Dataset* ds = dataset_from_arrays(X, y, n, nf, 1);
    if (ds) {
        ds->name = "iris";
        ds->num_classes = nc;
        ds->class_names = class_names;
    }
    free(X); free(y);
    return ds;
}

static Dataset* load_wine(void) {
    const char* path = cml_dataset_download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        "wine.data");
    if (!path) return NULL;

    float* X = NULL; float* y = NULL;
    int n = 0, nf = 0, nc = 0;
    if (cml_csv_parse(path, 0, &X, &y, &n, &nf, &nc, NULL) != 0)
        return NULL;

    Dataset* ds = dataset_from_arrays(X, y, n, nf, 1);
    if (ds) {
        ds->name = "wine";
        ds->num_classes = nc;
    }
    free(X); free(y);
    return ds;
}

static Dataset* load_breast_cancer(void) {
    const char* path = cml_dataset_download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        "wdbc.data");
    if (!path) return NULL;

    /* wdbc.data: ID(col0), Diagnosis(col1=M/B), 30 features(col2-31) */
    /* We need custom parsing: skip col0, col1 is target */
    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    /* Count lines */
    int n = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        if (strlen(line) > 5) n++;
    }
    rewind(f);

    float* X = malloc(sizeof(float) * n * 30);
    float* y = malloc(sizeof(float) * n);
    if (!X || !y) { free(X); free(y); fclose(f); return NULL; }

    int idx = 0;
    while (fgets(line, sizeof(line), f) && idx < n) {
        if (strlen(line) < 5) continue;

        /* Skip ID field */
        char* p = strchr(line, ',');
        if (!p) continue;
        p++;

        /* Read diagnosis: M=1, B=0 */
        y[idx] = (*p == 'M') ? 1.0f : 0.0f;
        p = strchr(p, ',');
        if (!p) continue;
        p++;

        /* Read 30 features */
        for (int j = 0; j < 30; j++) {
            X[idx * 30 + j] = (float)strtod(p, &p);
            if (*p == ',') p++;
        }
        idx++;
    }
    fclose(f);

    Dataset* ds = dataset_from_arrays(X, y, idx, 30, 1);
    if (ds) {
        ds->name = "breast_cancer";
        ds->num_classes = 2;
    }
    free(X); free(y);
    return ds;
}

static Dataset* load_boston(void) {
    const char* path = cml_dataset_download(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        "boston.csv");
    if (!path) return NULL;

    float* X = NULL; float* y = NULL;
    int n = 0, nf = 0, nc = 0;
    if (cml_csv_parse(path, -1, &X, &y, &n, &nf, &nc, NULL) != 0)
        return NULL;

    Dataset* ds = dataset_from_arrays(X, y, n, nf, 1);
    if (ds) {
        ds->name = "boston";
        ds->num_classes = 0; /* regression */
    }
    free(X); free(y);
    return ds;
}

static Dataset* load_mnist_dataset(const char* name, const char* base_url) {
    struct stat st;
    static char local_paths[4][256];
    const char* p_ti = NULL, *p_tl = NULL, *p_vi = NULL, *p_vl = NULL;

    /* First check local data/ directory (already present, no download needed) */
    const char* local_names[] = {
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
    };
    const char** ptrs[] = {&p_ti, &p_tl, &p_vi, &p_vl};

    for (int i = 0; i < 4; i++) {
        snprintf(local_paths[i], sizeof(local_paths[i]), "data/%s", local_names[i]);
        if (stat(local_paths[i], &st) == 0 && st.st_size > 0)
            *ptrs[i] = local_paths[i];
    }

    /* If local data not found, download from URL */
    if (!p_ti || !p_tl) {
        char url_train_img[256], url_train_lbl[256];
        char url_test_img[256], url_test_lbl[256];
        char gz_ti[128], gz_tl[128], gz_vi[128], gz_vl[128];
        char fn_ti[128], fn_tl[128], fn_vi[128], fn_vl[128];

        snprintf(url_train_img, sizeof(url_train_img), "%s/train-images-idx3-ubyte.gz", base_url);
        snprintf(url_train_lbl, sizeof(url_train_lbl), "%s/train-labels-idx1-ubyte.gz", base_url);
        snprintf(url_test_img, sizeof(url_test_img), "%s/t10k-images-idx3-ubyte.gz", base_url);
        snprintf(url_test_lbl, sizeof(url_test_lbl), "%s/t10k-labels-idx1-ubyte.gz", base_url);

        snprintf(gz_ti, sizeof(gz_ti), "%s-train-images-idx3-ubyte.gz", name);
        snprintf(gz_tl, sizeof(gz_tl), "%s-train-labels-idx1-ubyte.gz", name);
        snprintf(gz_vi, sizeof(gz_vi), "%s-t10k-images-idx3-ubyte.gz", name);
        snprintf(gz_vl, sizeof(gz_vl), "%s-t10k-labels-idx1-ubyte.gz", name);

        snprintf(fn_ti, sizeof(fn_ti), "%s-train-images-idx3-ubyte", name);
        snprintf(fn_tl, sizeof(fn_tl), "%s-train-labels-idx1-ubyte", name);
        snprintf(fn_vi, sizeof(fn_vi), "%s-t10k-images-idx3-ubyte", name);
        snprintf(fn_vl, sizeof(fn_vl), "%s-t10k-labels-idx1-ubyte", name);

        p_ti = download_and_gunzip(url_train_img, gz_ti, fn_ti);
        p_tl = download_and_gunzip(url_train_lbl, gz_tl, fn_tl);
        p_vi = download_and_gunzip(url_test_img, gz_vi, fn_vi);
        p_vl = download_and_gunzip(url_test_lbl, gz_vl, fn_vl);
    }

    if (!p_ti || !p_tl) {
        LOG_ERROR("[datasets] Failed to download %s training data", name);
        return NULL;
    }

    int n_train, rows, cols;
    float* train_img = cml_idx_load_images(p_ti, &n_train, &rows, &cols);
    int n_train_lbl;
    float* train_lbl = cml_idx_load_labels(p_tl, &n_train_lbl);

    if (!train_img || !train_lbl) {
        free(train_img); free(train_lbl);
        return NULL;
    }

    int total_n = n_train;
    int feat = rows * cols;

    float* all_X = NULL;
    float* all_y = NULL;

    if (p_vi && p_vl) {
        int n_test, tr, tc;
        float* test_img = cml_idx_load_images(p_vi, &n_test, &tr, &tc);
        int n_test_lbl;
        float* test_lbl = cml_idx_load_labels(p_vl, &n_test_lbl);

        if (test_img && test_lbl) {
            total_n = n_train + n_test;
            all_X = malloc(sizeof(float) * total_n * feat);
            all_y = malloc(sizeof(float) * total_n);
            memcpy(all_X, train_img, sizeof(float) * n_train * feat);
            memcpy(all_X + n_train * feat, test_img, sizeof(float) * n_test * feat);
            memcpy(all_y, train_lbl, sizeof(float) * n_train);
            memcpy(all_y + n_train, test_lbl, sizeof(float) * n_test);
        }
        free(test_img); free(test_lbl);
    }

    if (!all_X) {
        /* Only training data */
        all_X = train_img;
        all_y = train_lbl;
        train_img = NULL;
        train_lbl = NULL;
    }

    Dataset* ds = dataset_from_arrays(all_X, all_y, total_n, feat, 1);
    if (ds) {
        ds->name = name;
        ds->num_classes = 10;
    }

    if (all_X != train_img) { free(all_X); free(all_y); }
    free(train_img); free(train_lbl);
    return ds;
}

static Dataset* load_mnist(void) {
    return load_mnist_dataset("mnist", "https://storage.googleapis.com/cvdf-datasets/mnist");
}

static Dataset* load_fashion_mnist(void) {
    return load_mnist_dataset("fashion_mnist",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com");
}

static Dataset* load_cifar10(void) {
    const char* path = cml_dataset_download(
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        "cifar-10-binary.tar.gz");
    if (!path) return NULL;

    /* Extract */
    const char* dir = cml_dataset_cache_dir();
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "tar -xzf '%s' -C '%s' 2>/dev/null", path, dir);
    system(cmd);

    /* Read 5 training batches + 1 test batch */
    int total = 60000;
    int feat = 3072;
    float* X = malloc(sizeof(float) * total * feat);
    float* y = malloc(sizeof(float) * total);
    if (!X || !y) { free(X); free(y); return NULL; }

    int offset = 0;
    const char* batch_files[] = {
        "cifar-10-batches-bin/data_batch_1.bin",
        "cifar-10-batches-bin/data_batch_2.bin",
        "cifar-10-batches-bin/data_batch_3.bin",
        "cifar-10-batches-bin/data_batch_4.bin",
        "cifar-10-batches-bin/data_batch_5.bin",
        "cifar-10-batches-bin/test_batch.bin",
        NULL
    };

    for (int b = 0; batch_files[b]; b++) {
        char fpath[1024];
        snprintf(fpath, sizeof(fpath), "%s/%s", dir, batch_files[b]);
        FILE* f = fopen(fpath, "rb");
        if (!f) {
            LOG_WARNING("[datasets] Cannot open CIFAR batch: %s", fpath);
            continue;
        }

        /* Each record: 1 byte label + 3072 bytes pixel data */
        for (int i = 0; i < 10000 && offset < total; i++) {
            uint8_t label;
            if (fread(&label, 1, 1, f) != 1) break;
            y[offset] = (float)label;

            uint8_t pixels[3072];
            if (fread(pixels, 1, 3072, f) != 3072) break;
            for (int j = 0; j < 3072; j++)
                X[offset * feat + j] = pixels[j] / 255.0f;
            offset++;
        }
        fclose(f);
    }

    Dataset* ds = dataset_from_arrays(X, y, offset, feat, 1);
    if (ds) {
        ds->name = "cifar10";
        ds->num_classes = 10;
    }
    free(X); free(y);
    return ds;
}

static Dataset* load_airline(void) {
    int n;
    const float* data = cml_builtin_airline_data(&n);

    float* X = malloc(sizeof(float) * n);
    float* y = malloc(sizeof(float) * n);
    if (!X || !y) { free(X); free(y); return NULL; }

    memcpy(X, data, sizeof(float) * n);
    /* For time series: y is same as X (predict value) */
    memcpy(y, data, sizeof(float) * n);

    Dataset* ds = dataset_from_arrays(X, y, n, 1, 1);
    if (ds) {
        ds->name = "airline";
        ds->num_classes = 0;
    }
    free(X); free(y);
    return ds;
}

static Dataset* load_digits(void) {
    int n, feat;
    const float* data = cml_builtin_digits_data(&n, &feat);
    int nl;
    const float* labels = cml_builtin_digits_labels(&nl);

    /* Copy to mutable buffers for dataset_from_arrays */
    float* X = malloc(sizeof(float) * n * feat);
    float* y = malloc(sizeof(float) * n);
    if (!X || !y) { free(X); free(y); return NULL; }

    memcpy(X, data, sizeof(float) * n * feat);
    memcpy(y, labels, sizeof(float) * n);

    Dataset* ds = dataset_from_arrays(X, y, n, feat, 1);
    if (ds) {
        ds->name = "digits";
        ds->num_classes = 10;
    }
    free(X); free(y);
    return ds;
}

typedef Dataset* (*LoadFn)(void);

typedef struct {
    const char* name;
    LoadFn load;
} DatasetEntry;

static const DatasetEntry g_registry[] = {
    {"iris",          load_iris},
    {"wine",          load_wine},
    {"breast_cancer", load_breast_cancer},
    {"boston",         load_boston},
    {"mnist",         load_mnist},
    {"fashion_mnist", load_fashion_mnist},
    {"cifar10",       load_cifar10},
    {"airline",       load_airline},
    {"digits",        load_digits},
    {NULL, NULL}
};

Dataset* cml_dataset_load(const char* name) {
    if (!name) {
        LOG_ERROR("[datasets] NULL dataset name");
        return NULL;
    }

    for (int i = 0; g_registry[i].name; i++) {
        if (strcasecmp(name, g_registry[i].name) == 0) {
            LOG_INFO("[datasets] Loading '%s'...", name);
            Dataset* ds = g_registry[i].load();
            if (ds) {
                cml_dataset_compute_stats(ds);
                LOG_INFO("[datasets] Loaded '%s': %d samples, %d features",
                         name, ds->num_samples, ds->input_size);
            }
            return ds;
        }
    }

    LOG_ERROR("[datasets] Unknown dataset: '%s'", name);
    LOG_INFO("[datasets] Available: iris, wine, breast_cancer, boston, mnist, "
             "fashion_mnist, cifar10, airline, digits");
    return NULL;
}
