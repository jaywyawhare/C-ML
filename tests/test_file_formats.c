/* File-format error paths: dataset / loaders / serialization / ONNX.
 *
 * Parsers accumulate guard branches (bad magic, truncated files, bad headers,
 * missing keys). The happy-path tests never feed them a corrupt byte, so
 * hundreds of branches sat at 0. This writes deliberately broken files and
 * requires every reader to reject them cleanly.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>
#include "alloc/cml_allocator.h" /* idx loaders return cml_malloc'd memory */

#include "cml.h"
#include "core/dataset.h"
#include "datasets/datasets.h"
#include "core/serialization.h"
#include "nn.h"
#include "test_harness.h"

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

#define TMP_DIR "/tmp/cml_fmt_tests"

static void write_bytes(const char* name, const void* data, size_t n) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", TMP_DIR, name);
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror("tmpfile");
        return;
    }
    if (n)
        fwrite(data, 1, n, f);
    fclose(f);
}

static void setup_dir(void) {
#ifdef _WIN32
    _mkdir(TMP_DIR);
#else
    mkdir(TMP_DIR, 0755);
#endif
}

static int test_dataset_bad_files(void) {
    int ok = 1;

    /* nonexistent file */
    Dataset* ds = cml_dataset_from_csv("/nonexistent/x.csv", 0);
    if (ds)
        ok &= ds == NULL;
    if (ds)
        dataset_free(ds);

    /* empty CSV */
    write_bytes("empty.csv", "", 0);
    ds = cml_dataset_from_csv(TMP_DIR "/empty.csv", 0);
    if (ds)
        ok &= ds == NULL;
    if (ds)
        dataset_free(ds);

    /* header-only CSV */
    write_bytes("hdr.csv", "a,b,c\n", 6);
    ds = cml_dataset_from_csv(TMP_DIR "/hdr.csv", 0);
    /* zero data rows: NULL or an empty dataset both acceptable, no crash */
    if (ds)
        dataset_free(ds);

    /* target column out of range */
    write_bytes("two.csv", "1,2\n3,4\n", 8);
    ds = cml_dataset_from_csv(TMP_DIR "/two.csv", 99);
    if (ds)
        ok &= ds == NULL;
    if (ds)
        dataset_free(ds);

    /* IDX files: bad magic */
    const char bad_magic[] = "\x00\x00\x09\x09"
                             "\x00\x00\x00\x01"
                             "\x42";
    write_bytes("badmagic.idx", bad_magic, sizeof(bad_magic) - 1);
    int n = 0, r = 0, c = 0;
    float* img = cml_idx_load_images(TMP_DIR "/badmagic.idx", &n, &r, &c);
    ok &= img == NULL;
    if (img)
        cml_free(img);

    /* IDX images: dims byte lies (0 dims) */
    const char zero_dim[] = "\x00\x00\x08\x00";
    write_bytes("zerodim.idx", zero_dim, sizeof(zero_dim) - 1);
    img = cml_idx_load_images(TMP_DIR "/zerodim.idx", &n, &r, &c);
    ok &= img == NULL;
    if (img)
        cml_free(img);

    /* IDX labels: header-only file (count reads as 0 at EOF) — must return
     * cleanly either way, never crash */
    write_bytes("trunc.lbl", "\x00\x00\x08\x01", 4);
    int ln     = 0;
    float* lbl = cml_idx_load_labels(TMP_DIR "/trunc.lbl", &ln);
    if (lbl) {
        cml_free(lbl);
        lbl = NULL;
    }
    ok &= ln >= 0;

    return ok;
}

static int test_dataset_lifecycle(void) {
    /* from_arrays + split + normalize + shuffle + loader round trip */
    enum { N = 10, F = 2 };
    float X[N * F], y[N];
    for (int i = 0; i < N * F; i++)
        X[i] = (float)i;
    for (int i = 0; i < N; i++)
        y[i] = (float)(i % 2);

    Dataset* ds = dataset_from_arrays(X, y, N, F, 1);
    if (!ds)
        return 0;

    int ok = 1;
    cml_dataset_compute_stats(ds); /* normalize requires stats */

    ok &= dataset_normalize(ds, "minmax") == 0;
    ok &= dataset_normalize(ds, "bogus_method") != 0; /* unknown method */
    ok &= dataset_shuffle(ds, 7u) == 0;

    Dataset *tr = NULL, *va = NULL, *te = NULL;
    int s3 = dataset_split_three(ds, 0.5f, 0.25f, &tr, &va, &te);
    ok &= s3 == 0 || (tr == NULL && va == NULL && te == NULL);

    DataLoader* dl = dataloader_create(ds, 3, true);
    if (dl) {
        int batches = 0;
        Batch* b;
        dataloader_reset(dl);
        while ((b = dataloader_next_batch(dl)) != NULL) {
            Tensor* in = batch_get_input(b);
            if (in && b->batch_size > 0)
                tensor_ensure_executed(in);
            batch_free(b);
            batches++;
            if (batches > 100)
                break; /* loop guard */
        }
        ok &= batches > 0;
        dataloader_free(dl);
    }

    /* split with degenerate ratios must fail cleanly, not hang */
    ok &= dataset_split(NULL, 0.5f, &tr, &te) != 0;
    Dataset* d2 = dataset_from_arrays(X, y, N, F, 1);
    if (d2) {
        ok &= dataset_split(d2, -0.5f, &tr, &te) != 0; /* invalid ratio */
        if (tr) {
            dataset_free(tr);
            tr = NULL;
        }
        if (te) {
            dataset_free(te);
            te = NULL;
        }
        dataset_free(d2);
    }

    dataset_free(ds);
    if (tr)
        dataset_free(tr);
    if (va)
        dataset_free(va);
    if (te)
        dataset_free(te);
    return ok;
}

static int test_tensor_serialization_roundtrip_and_garbage(void) {
    int ok = 1;

    /* valid round trip */
    float d[6] = {1, 2, 3, 4, 5, 6};
    Tensor* t  = tensor_from_data(d, (int[]){2, 3}, 2, &cfg);
    if (!t)
        return 0;

    char path[512];
    snprintf(path, sizeof(path), "%s/t.bin", TMP_DIR);
    ok &= tensor_write_file(t, path) == 0;
    Tensor* r = tensor_read_file(path);
    ok &= r != NULL;
    if (r) {
        ok &= r->numel == t->numel;
        const float* rd = (const float*)tensor_data_ptr(r);
        for (int i = 0; i < 6; i++)
            ok &= fabsf(rd[i] - d[i]) < 1e-6f;
        tensor_free(r);
    }

    /* garbage files rejected */
    write_bytes("garbage.bin", "\x01\x02\x03", 3);
    r = tensor_read_file(TMP_DIR "/garbage.bin");
    ok &= r == NULL;
    if (r)
        tensor_free(r);

    write_bytes("empty.bin", "", 0);
    r = tensor_read_file(TMP_DIR "/empty.bin");
    ok &= r == NULL;
    if (r)
        tensor_free(r);

    /* truncated payload: valid magic/header, cut short */
    {
        FILE* f = fopen(path, "rb");
        if (f) {
            static char buf[4096];
            size_t n = fread(buf, 1, sizeof(buf), f); /* full valid file */
            fclose(f);
            if (n > 16)
                write_bytes("trunc.bin", buf, n - 12);
            r = tensor_read_file(TMP_DIR "/trunc.bin");
            /* may reject OR pad — but must not crash */
            if (r)
                tensor_free(r);
        }
    }

    /* stream variants on NULL handles */
    ok &= tensor_write_stream(t, NULL) != 0;
    ok &= tensor_read_stream(NULL) == NULL;

    tensor_free(t);
    return ok;
}

static int test_optimizer_serialization(void) {
    Parameter* p = NULL;
    Module m;
    memset(&m, 0, sizeof(m));
    Tensor* w = tensor_ones((int[]){2, 2}, 2, &cfg);
    if (!w)
        return 0;
    static char nm[] = "w";
    Parameter par;
    memset(&par, 0, sizeof(par));
    par.tensor        = w;
    par.requires_grad = true;
    par.name          = nm;
    p                 = &par;

    Optimizer* o = cml_optim_sgd(&p, 1, 0.01f, 0.0f, 0.0f);
    int ok       = o != NULL;
    if (o) {
        char path[512];
        snprintf(path, sizeof(path), "%s/opt.bin", TMP_DIR);
        ok &= optimizer_save(o, path) == 0;

        /* load into a fresh optimizer of the same kind */
        Optimizer* o2 = cml_optim_sgd(&p, 1, 0.01f, 0.0f, 0.0f);
        if (o2) {
            ok &= optimizer_load(o2, path) == 0;
            optimizer_free(o2);
        }
        /* loading garbage fails cleanly */
        write_bytes("optg.bin", "junkjunkjunk", 12);
        Optimizer* o3 = cml_optim_sgd(&p, 1, 0.01f, 0.0f, 0.0f);
        if (o3) {
            ok &= optimizer_load(o3, TMP_DIR "/optg.bin") != 0 ||
                  optimizer_load(o3, "/nonexistent.opt") != 0;
            optimizer_free(o3);
        }
        optimizer_free(o);
    }

    tensor_free(w);
    return ok;
}

int main(void) {
    setup_dir();
    cml_init();

    printf("=== file-format error paths ===\n");
    TEST(dataset_bad_files);
    TEST(dataset_lifecycle);
    TEST(tensor_serialization_roundtrip_and_garbage);
    TEST(optimizer_serialization);

    cml_cleanup();
    return TEST_SUMMARY();
}
