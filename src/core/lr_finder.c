#include "core/lr_finder.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"
#include "cml.h"
#include <math.h>
#include <string.h>

/** Free a learning-rate range-test result.
 * @param r (CMLLRFindResult*)
 * @return (void)
 */
void cml_lr_find_result_free(CMLLRFindResult* r) {
    if (!r)
        return;
    cml_free(r->lrs);
    cml_free(r->losses);
    cml_free(r);
}

/** Pull the next batch, resetting the loader once if it is exhausted.
 * @param loader (DataLoader*)
 * @return (Batch*) */
static Batch* next_wrapped(DataLoader* loader) {
    Batch* b = dataloader_next_batch(loader);
    if (!b) {
        dataloader_reset(loader);
        b = dataloader_next_batch(loader);
    }
    return b;
}

CMLLRFindResult* cml_lr_range_find(Module* model, DataLoader* loader, Optimizer* opt,
                                   Tensor* (*loss_fn)(Tensor*, Tensor*), float lr_min, float lr_max,
                                   int num_steps) {
    if (!model || !loader || !opt || !loss_fn || num_steps < 2 || lr_min <= 0.0f ||
        lr_max <= lr_min) {
        LOG_ERROR("cml_lr_range_find: invalid arguments");
        return NULL;
    }

    /* Snapshot parameters so the sweep leaves the model unchanged. */
    Parameter** params = NULL;
    int np             = 0;
    if (module_collect_parameters(model, &params, &np, true) != 0 || np <= 0) {
        LOG_ERROR("cml_lr_range_find: no parameters to train");
        cml_free(params);
        return NULL;
    }
    float** snap   = cml_calloc((size_t)np, sizeof(float*));
    size_t* snap_n = cml_calloc((size_t)np, sizeof(size_t));
    if (!snap || !snap_n) {
        cml_free(params);
        cml_free(snap);
        cml_free(snap_n);
        return NULL;
    }
    for (int i = 0; i < np; i++) {
        Tensor* t = params[i] ? params[i]->tensor : NULL;
        if (!t)
            continue;
        float* d = (float*)tensor_data_ptr(t);
        if (!d)
            continue;
        snap_n[i] = t->numel;
        snap[i]   = cml_malloc(t->numel * sizeof(float));
        if (snap[i])
            memcpy(snap[i], d, t->numel * sizeof(float));
    }

    float orig_lr = opt->num_param_groups > 0 ? opt->param_groups[0].lr : lr_min;

    CMLLRFindResult* r = cml_malloc(sizeof(CMLLRFindResult));
    float* lrs         = cml_malloc((size_t)num_steps * sizeof(float));
    float* losses      = cml_malloc((size_t)num_steps * sizeof(float));
    if (!r || !lrs || !losses) {
        cml_free(r);
        cml_free(lrs);
        cml_free(losses);
        r = NULL;
        goto restore;
    }

    int count   = 0;
    float ratio = lr_max / lr_min;
    for (int s = 0; s < num_steps; s++) {
        float lr = lr_min * powf(ratio, (float)s / (float)(num_steps - 1));
        optimizer_set_lr(opt, lr);

        Batch* b = next_wrapped(loader);
        if (!b || !b->X || !b->y) {
            if (b)
                batch_free(b);
            break;
        }
        Tensor* pred = module_forward(model, b->X);
        Tensor* loss = pred ? loss_fn(pred, b->y) : NULL;
        if (!loss) {
            batch_free(b);
            cml_reset_ir_context();
            break;
        }
        optimizer_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        optimizer_step(opt);

        lrs[count]    = lr;
        losses[count] = tensor_get_float(loss, 0);
        count++;

        batch_free(b);
        cml_reset_ir_context();
    }

    r->lrs    = lrs;
    r->losses = losses;
    r->count  = count;

    /* Suggested LR = steepest loss drop between consecutive points; fall back to the
     * LR at the minimum recorded loss. */
    if (count == 0) {
        r->suggested_lr = lr_min;
        goto restore;
    }
    float best_drop = 0.0f, min_loss = losses[0];
    int min_idx     = 0;
    r->suggested_lr = lrs[0];
    for (int i = 0; i < count; i++) {
        if (losses[i] < min_loss) {
            min_loss = losses[i];
            min_idx  = i;
        }
        if (i > 0) {
            float drop = losses[i - 1] - losses[i];
            if (drop > best_drop) {
                best_drop       = drop;
                r->suggested_lr = lrs[i];
            }
        }
    }
    if (best_drop <= 0.0f)
        r->suggested_lr = lrs[min_idx];

restore:
    for (int i = 0; i < np; i++) {
        Tensor* t = params[i] ? params[i]->tensor : NULL;
        if (!t || !snap[i])
            continue;
        float* d = (float*)tensor_data_ptr(t);
        if (d && snap_n[i] == t->numel)
            memcpy(d, snap[i], t->numel * sizeof(float));
        cml_free(snap[i]);
    }
    optimizer_set_lr(opt, orig_lr);
    cml_free(snap);
    cml_free(snap_n);
    cml_free(params);
    return (r && r->lrs) ? r : NULL;
}
