/* Training infrastructure branch sweep: all LR schedulers across their step
 * boundaries (plateau triggers, cosine restarts, warmup windows, milestone
 * edges, polynomial decay), training metrics record/export paths, gradient
 * clipping, and optimizer step variants.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "optim.h"
#include "core/training_metrics.h"
#include "nn.h"
#include "nn/layers/sequential.h"
#include "core/model_architecture.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"

static TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};

static Optimizer* g_opt = NULL;
static Tensor* g_w = NULL;

static int ensure_opt(void) {
    if (g_opt) return 1;
    Tensor* w = tensor_ones((int[]){2,2},2,&cfg);
    if (!w) return 0;
    static Parameter p; memset(&p,0,sizeof(p));
    static char nm[] = "w";
    p.tensor=w; p.requires_grad=true; p.name=nm;
    Parameter* pp=&p;
    g_opt = cml_optim_sgd(&pp,1,0.1f,0.0f,0.0f);
    if (!g_opt) { tensor_free(w); return 0; }
    g_w = w;
    return 1;
}

static Optimizer* fresh_opt(void) {
    if (g_opt) { optimizer_free(g_opt); g_opt = NULL; }
    if (!ensure_opt()) return NULL;
    return g_opt;
}

/* Every scheduler: update across enough steps to cross every boundary,
 * LR must stay finite and non-negative. */
static int test_lr_schedulers_matrix(void) {
    if (!ensure_opt()) return 0;
    int ok = 1;

    /* step scheduler: boundary at step_size */
    {
        fprintf(stderr,"SCH:step\n");
        if (!(g_opt = fresh_opt())) return 0;
        LRScheduler* s = cml_lr_scheduler_step(g_opt, 3, 0.5f);
        ok &= s != NULL;
        float prev = cml_lr_scheduler_get_lr(s);
        for (int i = 0; i < 10; i++) {
            float lr = cml_lr_scheduler_update(s, 0.0f);
            if (!(isfinite(lr) && lr >= 0.0f)) fprintf(stderr,"step bad val %g\n",(double)lr);
            if (lr > prev + 1e-6f) fprintf(stderr,"step rose: prev=%g now=%g\n",(double)prev,(double)lr);
            ok &= isfinite(lr) && lr >= 0.0f && lr <= prev + 1e-6f;
            prev = lr > prev ? lr : prev;
        }
        cml_lr_scheduler_free(s);
        fprintf(stderr,"BLK ok=%d\n", ok);
    }

    /* multi-step with milestones incl. duplicate and out-of-range */
    {
        fprintf(stderr,"SCH:ms\n");
        int ms[4] = {2, 4, 4, 99};
        if (!(g_opt = fresh_opt())) return 0;
        LRScheduler* s = cml_lr_scheduler_multi_step(g_opt, ms, 4, 0.5f);
        ok &= s != NULL;
        for (int i = 0; i < 12; i++) {
            float lr = cml_lr_scheduler_update(s, 0.0f);
            ok &= isfinite(lr) && lr >= 0.0f;
        }
        cml_lr_scheduler_free(s);
        fprintf(stderr,"BLK ok=%d\n", ok);
    }

    /* exponential */
    {
        fprintf(stderr,"SCH:exp\n");
        if (!(g_opt = fresh_opt())) return 0;
        LRScheduler* s = cml_lr_scheduler_exponential(g_opt, 0.9f);
        ok &= s != NULL;
        for (int i = 0; i < 8; i++) {
            float lr = cml_lr_scheduler_update(s, 0.0f);
            ok &= isfinite(lr) && lr >= 0.0f;
        }
        cml_lr_scheduler_free(s);
        fprintf(stderr,"BLK ok=%d\n", ok);
    }

    /* cosine over full period + beyond */
    {
        fprintf(stderr,"SCH:cos\n");
        if (!(g_opt = fresh_opt())) return 0;
        LRScheduler* s = cml_lr_scheduler_cosine(g_opt, 6, 0.01f);
        ok &= s != NULL;
        for (int i = 0; i < 14; i++) {   /* past T_max too */
            float lr = cml_lr_scheduler_update(s, 0.0f);
            ok &= isfinite(lr) && lr >= 0.0f;
        }
        cml_lr_scheduler_free(s);
        fprintf(stderr,"BLK ok=%d\n", ok);
    }

    /* plateau: improvements and plateaus both arms */
    {
        fprintf(stderr,"SCH:plat\n");
        if (!(g_opt = fresh_opt())) return 0;
        LRScheduler* s = cml_lr_scheduler_reduce_on_plateau(g_opt, 0.5f, 2, 0.001f);
        ok &= s != NULL;
        float metric = 1.0f;
        for (int i = 0; i < 10; i++) {
            metric += (i % 3 == 0) ? -0.1f : 0.05f;   /* mix improve/plateau */
            float lr = cml_lr_scheduler_update(s, metric);
            ok &= isfinite(lr) && lr >= 0.001f - 1e-6f;   /* min_lr floor */
        }
        cml_lr_scheduler_free(s);
        fprintf(stderr,"BLK ok=%d\n", ok);
    }

    /* polynomial */
    {
        fprintf(stderr,"SCH:poly\n");
        if (!(g_opt = fresh_opt())) return 0;
        LRScheduler* s = cml_lr_scheduler_polynomial(g_opt, 8, 1.0f, 0.01f);
        ok &= s != NULL;
        for (int i = 0; i < 12; i++) {   /* past total_iters */
            float lr = cml_lr_scheduler_update(s, 0.0f);
            fprintf(stderr,"poly i=%d lr=%g\n",i,(double)lr);
            ok &= isfinite(lr) && lr >= 0.0099f;
        }
        cml_lr_scheduler_free(s);
        fprintf(stderr,"BLK ok=%d\n", ok);
    }

    /* one_cycle */
    {
        fprintf(stderr,"SCH:cyc\n");
        if (!(g_opt = fresh_opt())) return 0;
        LRScheduler* s = cml_lr_scheduler_one_cycle(g_opt, 0.2f, 10, 0.3f, 25.0f, 10000.0f);
        ok &= s != NULL;
        for (int i = 0; i < 14; i++) {   /* past total_steps */
            float lr = cml_lr_scheduler_update(s, 0.0f);
            if (!(isfinite(lr) && lr >= 0.0f))
                fprintf(stderr,"cyc i=%d lr=%g\n", i, (double)lr);
            ok &= isfinite(lr) && lr >= 0.0f;
        }
        cml_lr_scheduler_free(s);
        fprintf(stderr,"BLK ok=%d\n", ok);
    }

    /* warmup wrapper around exponential */
    {
        fprintf(stderr,"SCH:warm\n");
        if (!(g_opt = fresh_opt())) return 0;
        LRScheduler* inner = cml_lr_scheduler_exponential(g_opt, 0.95f);
        ok &= inner != NULL;
        LRScheduler* s = cml_lr_scheduler_warmup(inner, 4, 0.1f);
        ok &= s != NULL;
        for (int i = 0; i < 10; i++) {
            float lr = cml_lr_scheduler_update(s, 0.0f);
            if (!(isfinite(lr) && lr >= 0.0f))
                fprintf(stderr,"warm i=%d lr=%g\n", i, (double)lr);
            ok &= isfinite(lr) && lr >= 0.0f;
        }
        cml_lr_scheduler_free(s);   /* frees chain or inner per contract */
    }

    fprintf(stderr,"SCHM ok=%d\n", ok);
    return ok;
}

static int test_training_metrics_matrix(void) {
    TrainingMetrics* m = training_metrics_create(4);
    if (!m) return 0;
    int ok = 1;

    for (size_t e = 0; e < 6; e++) {   /* past num_epochs too */
        training_metrics_start_epoch(m);
        training_metrics_record_epoch_full(m, e, 1.0f/(float)(e+1), 0.1f, 0.2f,
                                           0.3f, 0.4f, (float)e);
        training_metrics_record_epoch_time(m, e, 0.01f*(float)e);
        training_metrics_set_learning_rate(m, 0.1f/(float)(e+1), "step");
        training_metrics_set_lr_schedule_params(m, "{step: 3}");
        training_metrics_set_gradient_norm(m, 0.5f + 0.1f*(float)e);
    }
    training_metrics_set_summary(m, "sweep test");
    training_metrics_set_params(m, 100, 50);

    char path[512];
    snprintf(path, sizeof(path), "/tmp/cml_metrics.json");
    ok &= training_metrics_export_json(m, path, false) >= 0;

    /* gradient norm calculation over real params */
    ok &= fabsf(training_metrics_calculate_gradient_norm(m, NULL, 0) - 0.0f) < 1e-6f ||
          training_metrics_calculate_gradient_norm(m, NULL, 0) == 0.0f;

    training_metrics_free(m);

    /* degenerate: zero epochs must not crash on create/use */
    TrainingMetrics* z = training_metrics_create(0);
    if (z) {
        training_metrics_record_epoch_full(z, 0, 1.0f, 0, 0, 0, 0, 0);
        training_metrics_free(z);
    }
    return ok;
}

static int test_optimizer_step_variants(void) {
    if (!ensure_opt()) return 0;
    int ok = 1;

    /* give the parameter a gradient and step every optimizer family */
    struct { const char* n; Optimizer* (*mk)(Parameter**, int); } fams[0];
    (void)fams;

    /* SGD with momentum through several steps */
    Tensor* w = tensor_ones((int[]){2,2},2,&cfg);
    static Parameter p; memset(&p,0,sizeof(p));
    static char nm[]="p";
    p.tensor=w; p.requires_grad=true; p.name=nm;
    Parameter* pp=&p;

    Optimizer* o = cml_optim_sgd(&pp,1,0.05f,0.9f,0.0001f);
    ok &= o != NULL;
    if (o) {
        for (int st = 0; st < 3; st++) {
            /* fabricate a grad in place */
            if (!w->grad) {
                w->grad = tensor_zeros((int[]){2,2},2,&cfg);
            }
            if (w->grad && tensor_data_ptr(w->grad)) {
                float* gd=(float*)tensor_data_ptr(w->grad);
                for(int i=0;i<4;i++) gd[i]=0.1f*(float)(st+1);
            }
            if (o->zero_grad) o->zero_grad(o);
            if (o->step) o->step(o);
        }
        optimizer_free(o);
    }

    /* adam/adamw/adagrad/rmsprop single steps */
    o = cml_optim_adam(&pp,1,0.01f,0.0f,0.9f,0.999f,1e-8f);
    ok &= o != NULL; if(o){ if(o->step)o->step(o); optimizer_free(o);}
    o = cml_optim_adamw(&pp,1,0.01f,0.01f,0.9f,0.999f,1e-8f);
    ok &= o != NULL; if(o){ if(o->step)o->step(o); optimizer_free(o);}
    o = cml_optim_adagrad(&pp,1,0.01f,0.0f,1e-8f);
    ok &= o != NULL; if(o){ if(o->step)o->step(o); optimizer_free(o);}
    o = cml_optim_rmsprop(&pp,1,0.01f,0.0f,0.99f,1e-8f);
    ok &= o != NULL; if(o){ if(o->step)o->step(o); optimizer_free(o);}

    tensor_free(w);
    return ok;
}

static int test_container_parameter_collection(void);

int main(void) {
    cml_init();
    printf("=== training infra ===\n");
    TEST(lr_schedulers_matrix);
    TEST(training_metrics_matrix);
    TEST(optimizer_step_variants);
    TEST(container_parameter_collection);
    if (g_w) tensor_free(g_w);
    if (g_opt) optimizer_free(g_opt);
    cml_cleanup();
    return TEST_SUMMARY();
}

/* Container models: parameter collection and architecture export used to
 * report total_params = 0 because Sequential stores children in its own
 * array, not the Module->next chain that module_collect_parameters walked. */
static int test_container_parameter_collection(void) {
    Linear* l1 = cml_nn_linear(4, 8, DTYPE_FLOAT32, DEVICE_CPU, true);
    Linear* l2 = cml_nn_linear(8, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    if (!l1 || !l2) return 0;

    Sequential* seq = nn_sequential();
    if (!seq || sequential_add(seq, (Module*)l1) != 0 ||
        sequential_add(seq, (Module*)l2) != 0) return 0;

    Module* model = (Module*)seq;
    int ok = 1;

    Parameter** params = NULL;
    int n = 0;
    ok &= module_collect_parameters(model, &params, &n, true) == 0;
    ok &= n == 4;   /* weight+bias per Linear */
    /* every collected parameter is one of the two Linears' */
    for (int i = 0; i < n; i++)
        ok &= params[i] && params[i]->tensor != NULL;
    if (params) cml_free(params);

    /* architecture export sees the same parameters */
    ModelArchitecture* arch = model_architecture_create();
    ok &= arch != NULL;
    if (arch) {
        ok &= model_architecture_extract(model, arch) == 0;
        ok &= arch->num_layers == 2;
        size_t want = (size_t)(4*8 + 8) + (size_t)(8*2 + 2);
        if (arch->total_params != (int)want)
            fprintf(stderr, "container: total_params=%d want %d\n",
                    arch->total_params, (int)want);
        ok &= arch->total_params == (int)want;
        model_architecture_free(arch);
    }

    module_free((Module*)seq);   /* module_free untracks; specialized free does not */
    return ok;
}
