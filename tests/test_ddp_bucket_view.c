/* DDPConfig.gradient_as_bucket_view: gradients alias their slot in the flat
 * all-reduce bucket instead of being copied in and out around it.
 *
 * The risk the option carries is ownership, not arithmetic -- a gradient that
 * stops owning its data must still be freed exactly once, and must stay readable
 * after the DDP wrapper (which owns the buckets) is destroyed. These tests pin
 * the aliasing down at world_size 1, where it is fully observable: the alias is a
 * storage layout, so it does not need a second rank to exercise. Run under
 * -DENABLE_SANITIZERS=ON for the double-free half of the claim.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "nn/layers/linear.h"
#include "distributed/distributed.h"
#include "distributed/data_parallel.h"
#include "test_harness.h"
#include "test_require.h"

typedef struct {
    Linear* lin;
    Module* module;
    Tensor* input;
    Tensor* output;
} Fixture;

static Tensor* make_2d(const float* d, int rows, int cols) {
    int shape[2]     = {rows, cols};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    return tensor_from_data(d, shape, 2, &cfg);
}

/* Build a Linear with deterministic weights, run forward+backward so every
 * parameter has a gradient, and hand back the live module. */
static bool fixture_init(Fixture* f) {
    memset(f, 0, sizeof(*f));

    f->lin = nn_linear(4, 3, DTYPE_FLOAT32, DEVICE_CPU, true);
    if (!f->lin)
        return false;
    f->module = &f->lin->base;

    Tensor* w = f->lin->weight->tensor;
    REQUIRE(tensor_ensure_executed(w) == 0);
    float* wd = (float*)tensor_data_ptr(w);
    REQUIRE(wd);
    for (size_t i = 0; i < w->numel; i++)
        wd[i] = 0.2f + 0.03f * (float)(i % 6);

    const float x[8] = {1.0f, 0.5f, -0.25f, 2.0f, 0.75f, -1.5f, 0.125f, 1.0f};
    f->input         = make_2d(x, 2, 4);
    if (!f->input)
        return false;

    f->output = module_forward(f->module, f->input);
    if (!f->output)
        return false;

    const float seed[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 0.25f};
    Tensor* g           = make_2d(seed, 2, 3);
    REQUIRE(g);
    tensor_backward(f->output, g, false, false);
    tensor_free(g);

    return f->lin->weight->tensor->grad != NULL;
}

static void fixture_free(Fixture* f) {
    if (f->output)
        tensor_free(f->output);
    if (f->input)
        tensor_free(f->input);
    if (f->module)
        module_free(f->module);
    memset(f, 0, sizeof(*f));
}

/* Cumulative offset of param `idx` within its bucket. Bucket-view implies the
 * reserved layout, so every parameter counts whether or not it has a gradient. */
static size_t expected_offset(CMLDataParallel* ddp, int idx) {
    size_t off = 0;
    for (int i = 0; i < idx; i++)
        if (ddp->param_to_bucket[i] == ddp->param_to_bucket[idx] && ddp->all_params[i] &&
            ddp->all_params[i]->tensor)
            off += ddp->all_params[i]->tensor->numel;
    return off;
}

/* Snapshot every parameter's gradient into a flat buffer. Returns the count. */
static int snapshot_grads(CMLDataParallel* ddp, float* out, int cap) {
    int n = 0;
    for (int i = 0; i < ddp->num_params; i++) {
        Tensor* g = ddp->all_params[i]->tensor->grad;
        REQUIRE(g && tensor_ensure_executed(g) == 0);
        const float* gd = (const float*)tensor_data_ptr(g);
        REQUIRE(gd);
        for (size_t k = 0; k < g->numel; k++) {
            REQUIRE(n < cap);
            out[n++] = gd[k];
        }
    }
    return n;
}

/* The core claim: after a sync with the option on, each gradient's data pointer
 * IS its bucket slot, and the gradient no longer owns that memory. */
static bool test_grads_alias_bucket_slots(void) {
    REQUIRE(cml_dist_init(DIST_BACKEND_GLOO, 1, 0) == 0);

    Fixture f;
    if (!fixture_init(&f)) {
        fixture_free(&f);
        cml_dist_destroy();
        return false;
    }

    DDPConfig cfg               = cml_ddp_default_config();
    cfg.gradient_as_bucket_view = 1;
    CMLDataParallel* ddp        = cml_ddp_create(f.module, &cfg);
    REQUIRE(ddp && ddp->grad_is_view);

    bool ok = cml_ddp_sync_gradients(ddp) == 0;

    int aliased = 0;
    for (int i = 0; ok && i < ddp->num_params; i++) {
        Tensor* g = ddp->all_params[i]->tensor->grad;
        if (!g)
            continue;
        if (!ddp->grad_is_view[i])
            continue;

        int b       = ddp->param_to_bucket[i];
        float* slot = ddp->buckets[b] + expected_offset(ddp, i);
        if ((float*)g->data != slot) {
            printf("(param %d not at slot) ", i);
            ok = false;
        } else if (g->owns_data) {
            printf("(param %d still owns data) ", i);
            ok = false;
        } else {
            aliased++;
        }
    }

    /* A run where nothing got aliased would satisfy the loop vacuously. */
    if (ok && aliased != ddp->num_params) {
        printf("(aliased %d of %d) ", aliased, ddp->num_params);
        ok = false;
    }

    cml_ddp_free(ddp);
    fixture_free(&f);
    cml_dist_destroy();
    return ok;
}

/* Without the option the gradients must keep their own storage -- a guard
 * against the aliasing leaking into the default path. */
static bool test_default_config_does_not_alias(void) {
    REQUIRE(cml_dist_init(DIST_BACKEND_GLOO, 1, 0) == 0);

    Fixture f;
    if (!fixture_init(&f)) {
        fixture_free(&f);
        cml_dist_destroy();
        return false;
    }

    CMLDataParallel* ddp = cml_ddp_create(f.module, NULL);
    REQUIRE(ddp);

    bool ok = ddp->grad_is_view == NULL && cml_ddp_sync_gradients(ddp) == 0;

    for (int i = 0; ok && i < ddp->num_params; i++) {
        Tensor* g = ddp->all_params[i]->tensor->grad;
        if (!g || !g->data)
            continue;
        for (int b = 0; b < ddp->num_buckets; b++) {
            if (!ddp->buckets[b])
                continue;
            float* lo = ddp->buckets[b];
            float* hi = lo + ddp->bucket_sizes[b];
            if ((float*)g->data >= lo && (float*)g->data < hi) {
                printf("(param %d aliased by default) ", i);
                ok = false;
            }
        }
    }

    cml_ddp_free(ddp);
    fixture_free(&f);
    cml_dist_destroy();
    return ok;
}

/* Aliasing must not disturb the values: the gradient read back through the view
 * has to match what the option-off path produces. */
static bool test_alias_preserves_gradient_values(void) {
    float base[64], viewed[64];
    int n_base = 0, n_view = 0;

    REQUIRE(cml_dist_init(DIST_BACKEND_GLOO, 1, 0) == 0);
    Fixture f;
    if (!fixture_init(&f)) {
        fixture_free(&f);
        cml_dist_destroy();
        return false;
    }
    CMLDataParallel* d0 = cml_ddp_create(f.module, NULL);
    REQUIRE(d0 && cml_ddp_sync_gradients(d0) == 0);
    n_base = snapshot_grads(d0, base, 64);
    cml_ddp_free(d0);
    fixture_free(&f);
    cml_dist_destroy();

    REQUIRE(cml_dist_init(DIST_BACKEND_GLOO, 1, 0) == 0);
    if (!fixture_init(&f)) {
        fixture_free(&f);
        cml_dist_destroy();
        return false;
    }
    DDPConfig cfg               = cml_ddp_default_config();
    cfg.gradient_as_bucket_view = 1;
    CMLDataParallel* d1         = cml_ddp_create(f.module, &cfg);
    REQUIRE(d1 && cml_ddp_sync_gradients(d1) == 0);
    n_view = snapshot_grads(d1, viewed, 64);
    cml_ddp_free(d1);
    fixture_free(&f);
    cml_dist_destroy();

    if (n_base != n_view || n_base == 0) {
        printf("(count %d vs %d) ", n_base, n_view);
        return false;
    }
    for (int i = 0; i < n_base; i++) {
        if (base[i] != viewed[i]) {
            printf("(grad[%d] %g vs %g) ", i, (double)base[i], (double)viewed[i]);
            return false;
        }
    }
    return true;
}

/* cml_ddp_free releases the buckets the gradients were pointing into, so it must
 * hand each aliased gradient its own storage back first -- otherwise reading a
 * gradient after teardown is a use-after-free, and freeing the module is a
 * double-free. */
static bool test_grads_survive_ddp_free(void) {
    REQUIRE(cml_dist_init(DIST_BACKEND_GLOO, 1, 0) == 0);

    Fixture f;
    if (!fixture_init(&f)) {
        fixture_free(&f);
        cml_dist_destroy();
        return false;
    }

    DDPConfig cfg               = cml_ddp_default_config();
    cfg.gradient_as_bucket_view = 1;
    CMLDataParallel* ddp        = cml_ddp_create(f.module, &cfg);
    REQUIRE(ddp && cml_ddp_sync_gradients(ddp) == 0);

    float before[64];
    int n = snapshot_grads(ddp, before, 64);
    REQUIRE(n > 0);

    /* Capture the parameter list: it is owned by ddp and gone after the free. */
    int num_params   = ddp->num_params;
    Tensor** tensors = malloc((size_t)num_params * sizeof(Tensor*));
    REQUIRE(tensors);
    for (int i = 0; i < num_params; i++)
        tensors[i] = ddp->all_params[i]->tensor;

    cml_ddp_free(ddp); /* buckets released here */

    bool ok = true;
    int k   = 0;
    for (int i = 0; i < num_params && ok; i++) {
        Tensor* g = tensors[i]->grad;
        if (!g || !g->data) {
            printf("(param %d lost its grad) ", i);
            ok = false;
            break;
        }
        if (!g->owns_data) {
            printf("(param %d still borrows freed bucket) ", i);
            ok = false;
            break;
        }
        const float* gd = (const float*)g->data;
        for (size_t j = 0; j < g->numel; j++) {
            if (gd[j] != before[k++]) {
                printf("(param %d value changed) ", i);
                ok = false;
                break;
            }
        }
    }

    free(tensors);
    fixture_free(&f); /* frees the module, and with it the gradients */
    cml_dist_destroy();
    return ok;
}

/* Re-syncing must be idempotent: the second pass sees gradients that already
 * alias their slots and must leave both the pointers and the values alone. */
static bool test_repeated_sync_is_stable(void) {
    REQUIRE(cml_dist_init(DIST_BACKEND_GLOO, 1, 0) == 0);

    Fixture f;
    if (!fixture_init(&f)) {
        fixture_free(&f);
        cml_dist_destroy();
        return false;
    }

    DDPConfig cfg               = cml_ddp_default_config();
    cfg.gradient_as_bucket_view = 1;
    CMLDataParallel* ddp        = cml_ddp_create(f.module, &cfg);
    REQUIRE(ddp);

    float first[64], second[64];
    REQUIRE(cml_ddp_sync_gradients(ddp) == 0);
    int n1 = snapshot_grads(ddp, first, 64);

    void** ptrs = malloc((size_t)ddp->num_params * sizeof(void*));
    REQUIRE(ptrs);
    for (int i = 0; i < ddp->num_params; i++)
        ptrs[i] = ddp->all_params[i]->tensor->grad->data;

    REQUIRE(cml_ddp_sync_gradients(ddp) == 0);
    int n2 = snapshot_grads(ddp, second, 64);

    bool ok = (n1 == n2) && n1 > 0;
    for (int i = 0; ok && i < ddp->num_params; i++) {
        if (ddp->all_params[i]->tensor->grad->data != ptrs[i]) {
            printf("(param %d repointed) ", i);
            ok = false;
        }
    }
    for (int i = 0; ok && i < n1; i++) {
        if (first[i] != second[i]) {
            printf("(grad[%d] drifted %g -> %g) ", i, (double)first[i], (double)second[i]);
            ok = false;
        }
    }

    free(ptrs);
    cml_ddp_free(ddp);
    fixture_free(&f);
    cml_dist_destroy();
    return ok;
}

/* The option no longer advertises itself as inert. */
static bool test_default_config_leaves_option_off(void) {
    DDPConfig cfg = cml_ddp_default_config();
    return cfg.gradient_as_bucket_view == 0;
}

int main(void) {
    printf("DDP gradient_as_bucket_view Tests\n\n");

    TEST(default_config_leaves_option_off);
    TEST(default_config_does_not_alias);
    TEST(grads_alias_bucket_slots);
    TEST(alias_preserves_gradient_values);
    TEST(repeated_sync_is_stable);
    TEST(grads_survive_ddp_free);

    return TEST_SUMMARY();
}
