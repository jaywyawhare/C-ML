/* Smoke + correctness test for training_metrics.c (previously untested): create,
 * record epochs, export JSON, and read it back. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cml.h"
#include "test_harness.h"

int main(void) {
    cml_init();
    printf("=== training_metrics ===\n");

    int N              = 5;
    TrainingMetrics* m = training_metrics_create((size_t)N);
    CHECK("training_metrics_create", m != NULL);
    if (!m) {
        printf("\nTRAINING METRICS FAILED\n");
        return 1;
    }

    /* record a decreasing loss / increasing accuracy curve */
    for (int e = 0; e < N; e++)
        training_metrics_record_epoch(m, (size_t)e, 1.0f / (float)(e + 1), 0.5f + 0.08f * e);

    const char* path = "/tmp/cml_metrics_test.json";
    remove(path);
    int rc = training_metrics_export_json(m, path, NULL);
    CHECK("export_json returns success", rc == 0);

    /* read the JSON back and confirm it captured a recorded value */
    FILE* f = fopen(path, "r");
    CHECK("export_json produced a file", f != NULL);
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* buf  = (char*)malloc((size_t)sz + 1);
        size_t got = fread(buf, 1, (size_t)sz, f);
        buf[got]   = '\0';
        fclose(f);
        CHECK("JSON file is non-empty", sz > 0);
        /* first epoch loss was 1.0 - the exported JSON should mention loss */
        CHECK("JSON contains a 'loss' field", strstr(buf, "loss") != NULL);
        free(buf);
        remove(path);
    }

    /* ── Distribution capture ────────────────────────────────────────────
       A known population: weights are the constant 2.0, gradients run -1..+1
       across a linear layer, so every band has a value we can predict. */
    {
        Linear* fc = cml_nn_linear(4, 3, DTYPE_FLOAT32, DEVICE_CPU, false);
        CHECK("linear layer for distribution test", fc != NULL);

        Parameter** params = NULL;
        int nparams        = 0;
        module_collect_parameters((Module*)fc, &params, &nparams, true);
        CHECK("collected parameters", params != NULL && nparams > 0);

        if (params && nparams > 0) {
            Tensor* w = params[0]->tensor;
            size_t n  = 0;
            float* wd = cml_tensor_float_buffer(w, &n);
            CHECK("weight buffer resolved", wd != NULL && n > 0);

            if (wd && n) {
                for (size_t i = 0; i < n; i++)
                    wd[i] = 2.0f;

                /* Give the parameter a gradient spanning -1..+1. */
                int gshape[8];
                for (int d = 0; d < w->ndim; d++)
                    gshape[d] = w->shape[d];
                TensorConfig gcfg = {.dtype      = DTYPE_FLOAT32,
                                     .device     = DEVICE_CPU,
                                     .has_dtype  = true,
                                     .has_device = true};
                Tensor* g         = cml_zeros(gshape, w->ndim, &gcfg);
                size_t gn         = 0;
                float* gd         = cml_tensor_float_buffer(g, &gn);
                if (gd) {
                    for (size_t i = 0; i < gn; i++)
                        gd[i] = -1.0f + 2.0f * ((float)i / (float)(gn > 1 ? gn - 1 : 1));
                }
                w->grad = g;

                training_metrics_record_distributions(m, 0, (void**)params, nparams);

                const DistributionSummary* wdist = &m->epoch_weight_dist[0];
                const DistributionSummary* gdist = &m->epoch_grad_dist[0];

                CHECK("weight min == max == 2.0 (constant population)",
                      wdist->min == 2.0f && wdist->max == 2.0f);
                CHECK("weight median tracks the constant", wdist->p50 == 2.0f);
                CHECK("weight std is zero", wdist->std < 1e-6f);
                CHECK("gradient spans -1..+1", gdist->min <= -0.99f && gdist->max >= 0.99f);
                CHECK("gradient median near 0", gdist->p50 > -0.15f && gdist->p50 < 0.15f);
                CHECK("gradient quartiles ordered",
                      gdist->p25 <= gdist->p50 && gdist->p50 <= gdist->p75);
                CHECK("distributions marked present", m->has_distributions);
            }
            cml_free(params);
        }
    }

    /* Re-export now that distributions exist and confirm they reach the JSON. */
    {
        const char* dpath = "/tmp/cml_metrics_dist_test.json";
        remove(dpath);
        CHECK("export with distributions", training_metrics_export_json(m, dpath, NULL) == 0);
        FILE* df = fopen(dpath, "r");
        if (df) {
            fseek(df, 0, SEEK_END);
            long dsz = ftell(df);
            fseek(df, 0, SEEK_SET);
            char* dbuf  = (char*)malloc((size_t)dsz + 1);
            size_t dgot = fread(dbuf, 1, (size_t)dsz, df);
            dbuf[dgot]  = '\0';
            fclose(df);
            CHECK("JSON has grad_distribution", strstr(dbuf, "grad_distribution") != NULL);
            CHECK("JSON has weight_distribution", strstr(dbuf, "weight_distribution") != NULL);
            CHECK("JSON has validation series", strstr(dbuf, "epoch_validation_losses") != NULL);
            CHECK("distribution carries percentile bands", strstr(dbuf, "\"p75\"") != NULL);
            free(dbuf);
            remove(dpath);
        }
    }

    /* Distributions read each parameter in its own dtype: a half-precision buffer read
     * as float32 overran it and summarised garbage. */
    {
        TensorConfig hcfg = {
            .dtype = DTYPE_FLOAT16, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
        Tensor* h       = tensor_empty((int[]){4}, 1, &hcfg);
        const float v[] = {-2.0f, 0.5f, 1.0f, 3.0f};
        for (int i = 0; h && i < 4; i++)
            tensor_set_float(h, (size_t)i, v[i]);
        Parameter p    = {.tensor = h};
        void* params[] = {&p};
        training_metrics_record_distributions(m, 1, params, 1);
        CHECK("half-precision weights summarised by value",
              m->epoch_weight_dist[1].min == -2.0f && m->epoch_weight_dist[1].max == 3.0f);
        tensor_free(h);
    }

    /* A classification loss stashes its prediction for the next backward's accuracy. One
     * that is never backpropagated (a validation loss) must not leave that stash dangling
     * once its tensors are freed. */
    {
        float probs[2] = {0.2f, 0.9f}, labels[2] = {0.0f, 1.0f};
        Tensor* pred = cml_tensor_1d(probs, 2);
        Tensor* tgt  = cml_tensor_1d(labels, 2);
        Tensor* bce  = tensor_bce_loss(pred, tgt);
        tensor_free(bce);
        tensor_free(pred);
        tensor_free(tgt);
        cml_ir_reset_global_context();

        float xv[2] = {1.0f, 2.0f}, yv[2] = {0.0f, 0.0f};
        Tensor* x        = cml_tensor_1d(xv, 2);
        Tensor* y        = cml_tensor_1d(yv, 2);
        x->requires_grad = true;
        Tensor* mse      = tensor_mse_loss(x, y);
        tensor_backward(mse, NULL, false, false);
        CHECK("backward after an unpropagated classification loss", x->grad != NULL);
        tensor_free(mse);
        tensor_free(x);
        tensor_free(y);
        cml_ir_reset_global_context();
    }

    training_metrics_free(m);
    CHECK("training_metrics_free (no crash)", 1);

    return TEST_SUMMARY();
}
