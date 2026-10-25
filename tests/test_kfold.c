/* K-fold and stratified k-fold splitting: every sample appears in exactly one
 * validation fold across the k folds, train and val are disjoint and sum to the whole,
 * and stratified folds preserve class balance. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cml.h"
#include "core/dataset.h"
#include "tensor/tensor.h"
#include "test_harness.h"

/* n samples, 1 feature = index, label = i % num_classes. */
static Dataset* make_ds(int n, int num_classes) {
    float* X = malloc(sizeof(float) * n);
    float* y = malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) {
        X[i] = (float)i;
        y[i] = (float)(i % num_classes);
    }
    Dataset* d = dataset_from_arrays(X, y, n, 1, 1);
    if (d)
        d->num_classes = num_classes;
    free(X);
    free(y);
    return d;
}

static int test_kfold_partitions(void) {
    int n = 20, k = 5;
    Dataset* ds  = make_ds(n, 2);
    int seen[20] = {0};
    int ok       = ds != NULL;
    for (int f = 0; f < k && ok; f++) {
        Dataset *tr = NULL, *va = NULL;
        ok &= dataset_kfold(ds, k, f, false, 0, &tr, &va) == 0;
        ok &= tr && va && (tr->num_samples + va->num_samples == n);
        /* validation rows are the contiguous fold [f*4, f*4+4) */
        const float* vx = (const float*)tensor_data_ptr(va->X);
        for (int i = 0; i < va->num_samples; i++)
            seen[(int)(vx[i] + 0.5f)]++;
        ok &= va->num_samples == n / k;
        dataset_free(tr);
        dataset_free(va);
    }
    for (int i = 0; i < n; i++)
        ok &= seen[i] == 1; /* each sample validated exactly once */
    dataset_free(ds);
    return ok;
}

static int test_kfold_shuffle_deterministic(void) {
    Dataset* ds = make_ds(20, 2);
    Dataset *t1 = NULL, *v1 = NULL, *t2 = NULL, *v2 = NULL;
    int ok = dataset_kfold(ds, 4, 0, true, 42, &t1, &v1) == 0;
    ok &= dataset_kfold(ds, 4, 0, true, 42, &t2, &v2) == 0;
    /* same seed -> identical validation rows */
    const float* a = v1 ? (const float*)tensor_data_ptr(v1->X) : NULL;
    const float* b = v2 ? (const float*)tensor_data_ptr(v2->X) : NULL;
    ok &= a && b && v1->num_samples == v2->num_samples;
    for (int i = 0; ok && i < v1->num_samples; i++)
        ok &= a[i] == b[i];
    dataset_free(t1);
    dataset_free(v1);
    dataset_free(t2);
    dataset_free(v2);
    dataset_free(ds);
    return ok;
}

static int test_stratified_balance(void) {
    int n = 30, k = 3, nc = 3;
    Dataset* ds   = make_ds(n, nc); /* 10 per class */
    int ok        = ds != NULL;
    int total_val = 0;
    for (int f = 0; f < k && ok; f++) {
        Dataset *tr = NULL, *va = NULL;
        ok &= dataset_kfold_stratified(ds, k, f, 7, &tr, &va) == 0;
        /* each fold should hold ~10/3 of each class; with 10 per class and k=3
         * every fold has 3 or 4 of each class. */
        int per_class[3] = {0, 0, 0};
        const float* vy  = (const float*)tensor_data_ptr(va->y);
        for (int i = 0; i < va->num_samples; i++)
            per_class[(int)(vy[i] + 0.5f)]++;
        for (int c = 0; c < nc; c++)
            ok &= (per_class[c] == 3 || per_class[c] == 4);
        total_val += va->num_samples;
        dataset_free(tr);
        dataset_free(va);
    }
    ok &= total_val == n; /* folds partition the set */
    dataset_free(ds);
    return ok;
}

static int test_rejects_bad_args(void) {
    Dataset* ds = make_ds(10, 2);
    Dataset *tr = NULL, *va = NULL;
    int ok = dataset_kfold(ds, 1, 0, false, 0, &tr, &va) == -1; /* k < 2 */
    ok &= dataset_kfold(ds, 3, 3, false, 0, &tr, &va) == -1;    /* fold == k */
    dataset_free(ds);
    return ok;
}

int main(void) {
    cml_init();
    printf("=== k-fold cross-validation ===\n");
    TEST(kfold_partitions);
    TEST(kfold_shuffle_deterministic);
    TEST(stratified_balance);
    TEST(rejects_bad_args);
    cml_cleanup();
    return TEST_SUMMARY();
}
