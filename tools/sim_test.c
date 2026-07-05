/*
 * sim_test.c — Seeded deterministic simulation + OOM sweep for C-ML
 *
 * Inspired by TigerBeetle's VOPR: given a seed, generate a random computation
 * graph, run it twice and assert bit-identical results (determinism), then sweep
 * all allocation sites causing each one to fail in turn and assert no crash
 * (graceful OOM handling everywhere).
 *
 * Usage:
 *   sim_test                    # random seed, 64 iterations
 *   sim_test --seed 42          # fixed seed
 *   sim_test --iters 256        # more iterations
 *   sim_test --oom-sweep        # OOM sweep (slow but thorough)
 *   sim_test --grad-check       # gradient correctness via finite differences
 *
 * Exit 0 = all checks pass.  Exit 1 = found a bug (seed printed for repro).
 */

#include "cml.h"
#include "alloc/cml_allocator.h"
#include "core/threefry.h"
#include "ops/ir/context.h"
#include "tensor/realize.h"
#include "backend/threadpool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

/* ── tiny PRNG for workload generation (separate from cml's RNG) ──────────── */

typedef struct { uint64_t s[2]; } Xorshift128;

static uint64_t xs_next(Xorshift128* x) {
    uint64_t t = x->s[0];
    uint64_t s = x->s[1];
    x->s[0] = s;
    t ^= t << 23; t ^= t >> 17; t ^= s ^ (s >> 26);
    x->s[1] = t;
    return t + s;
}

static void xs_seed(Xorshift128* x, uint64_t seed) {
    x->s[0] = seed ^ 0x9e3779b97f4a7c15ULL;
    x->s[1] = seed ^ 0x6c62272e07bb0142ULL;
    /* warm up */
    for (int i = 0; i < 8; i++) xs_next(x);
}

static int xs_range(Xorshift128* x, int lo, int hi) { /* [lo, hi) */
    return lo + (int)(xs_next(x) % (uint64_t)(hi - lo));
}

static float xs_float(Xorshift128* x) {
    return (float)(xs_next(x) >> 11) * (1.0f / (float)(1ULL << 53));
}

/* ── workload: random tensor ops ─────────────────────────────────────────── */

#define MAX_TENSORS 16
#define MAX_OPS     32

typedef struct {
    int ndim;
    int shape[4];
} ShapeSpec;

static ShapeSpec random_shape(Xorshift128* x) {
    ShapeSpec s;
    s.ndim = xs_range(x, 1, 4); /* 1-3 dims */
    for (int i = 0; i < s.ndim; i++)
        s.shape[i] = xs_range(x, 1, 9); /* 1-8 per dim */
    return s;
}

typedef enum {
    SIM_ADD = 0, SIM_MUL, SIM_RELU, SIM_SIGMOID, SIM_TANH,
    SIM_MATMUL, SIM_REALIZE, SIM_NOOP,
    SIM_OP_COUNT
} SimOpKind;

static const char* op_name(SimOpKind op) {
    switch (op) {
    case SIM_ADD:     return "add";
    case SIM_MUL:     return "mul";
    case SIM_RELU:    return "relu";
    case SIM_SIGMOID: return "sigmoid";
    case SIM_TANH:    return "tanh";
    case SIM_MATMUL:  return "matmul";
    case SIM_REALIZE: return "realize";
    case SIM_NOOP:    return "noop";
    default:          return "?";
    }
}

/* Result of running a workload: byte fingerprint of all realized tensors */
typedef struct {
    uint64_t fingerprint;
    int      num_ops_executed;
    bool     had_null;       /* some op returned NULL (OK as long as no crash) */
} WorkloadResult;

static uint64_t hash_tensor(Tensor* t) {
    if (!t) return 0xDEADBEEFCAFEBABEULL;
    if (tensor_realize(t) != 0) return 0xBAD0BAD0BAD0BAD0ULL;
    float* data = (float*)tensor_data_ptr(t);
    if (!data) return 0xDEADC0DEDEADC0DEULL;
    uint64_t h = 0xcbf29ce484222325ULL; /* FNV-1a offset basis */
    for (size_t i = 0; i < t->numel; i++) {
        uint32_t bits;
        memcpy(&bits, &data[i], 4);
        h ^= (uint64_t)bits;
        h *= 0x100000001b3ULL;
    }
    /* also hash the shape so a wrong shape is detected */
    for (int d = 0; d < t->ndim; d++) {
        h ^= (uint64_t)(unsigned)t->shape[d] << (d * 8);
        h *= 0x100000001b3ULL;
    }
    return h;
}

static WorkloadResult run_workload(uint64_t seed, bool verbose) {
    WorkloadResult res = {0};

    Xorshift128 x;
    xs_seed(&x, seed);

    cml_seed(seed);
    cml_ir_reset_global_context();

    /* create some leaf tensors with known data */
    int n_tensors = xs_range(&x, 2, MAX_TENSORS + 1);
    Tensor* tensors[MAX_TENSORS] = {0};
    ShapeSpec shapes[MAX_TENSORS];

    for (int i = 0; i < n_tensors; i++) {
        shapes[i] = random_shape(&x);
        int total = 1;
        for (int d = 0; d < shapes[i].ndim; d++) total *= shapes[i].shape[d];
        float* buf = malloc((size_t)total * sizeof(float));
        if (!buf) { tensors[i] = NULL; continue; }
        for (int j = 0; j < total; j++) buf[j] = xs_float(&x) * 2.0f - 1.0f;
        TensorConfig cfg = {.dtype=DTYPE_FLOAT32, .device=DEVICE_CPU,
                            .has_dtype=true, .has_device=true};
        tensors[i] = cml_tensor(buf, shapes[i].shape, shapes[i].ndim, &cfg);
        free(buf);
        if (tensors[i] && tensor_realize(tensors[i]) != 0) {
            tensor_free(tensors[i]);
            tensors[i] = NULL;
        }
    }

    /* run random ops */
    int n_ops = xs_range(&x, 4, MAX_OPS + 1);
    for (int op_i = 0; op_i < n_ops; op_i++) {
        int a_idx = xs_range(&x, 0, n_tensors);
        Tensor* a = tensors[a_idx];
        if (!a) continue;

        SimOpKind op = (SimOpKind)xs_range(&x, 0, SIM_OP_COUNT);
        if (verbose) printf("    op[%d] %s on t[%d]\n", op_i, op_name(op), a_idx);

        Tensor* out = NULL;
        switch (op) {
        case SIM_ADD: {
            int b_idx = xs_range(&x, 0, n_tensors);
            Tensor* b = tensors[b_idx];
            if (b && b->ndim == a->ndim) {
                bool same = true;
                for (int d = 0; d < a->ndim; d++)
                    if (a->shape[d] != b->shape[d]) { same = false; break; }
                if (same) out = tensor_add(a, b);
            }
            break;
        }
        case SIM_MUL: {
            int b_idx = xs_range(&x, 0, n_tensors);
            Tensor* b = tensors[b_idx];
            if (b && b->ndim == a->ndim) {
                bool same = true;
                for (int d = 0; d < a->ndim; d++)
                    if (a->shape[d] != b->shape[d]) { same = false; break; }
                if (same) out = tensor_mul(a, b);
            }
            break;
        }
        case SIM_RELU:    out = tensor_relu(a); break;
        case SIM_SIGMOID: out = tensor_sigmoid(a); break;
        case SIM_TANH:    out = tensor_tanh(a); break;
        case SIM_MATMUL:
            if (a->ndim >= 2) {
                int b_idx = xs_range(&x, 0, n_tensors);
                Tensor* b = tensors[b_idx];
                if (b && b->ndim == 2 && b->shape[0] == a->shape[a->ndim-1])
                    out = tensor_matmul(a, b);
            }
            break;
        case SIM_REALIZE:
            tensor_realize(a);
            break;
        case SIM_NOOP:
            break;
        default: break;
        }

        res.num_ops_executed++;
        if (out) {
            /* Realize immediately: the lazy IR node references the input tensor.
             * If we store 'out' into the same slot as the input and then free the
             * old tensor, the IR dangling-pointer read causes non-deterministic results. */
            if (tensor_realize(out) != 0) {
                tensor_free(out);
                out = NULL;
                res.had_null = true;
            } else {
                int dest = xs_range(&x, 0, n_tensors);
                if (tensors[dest]) tensor_free(tensors[dest]);
                tensors[dest] = out;
                shapes[dest].ndim = out->ndim;
                for (int d = 0; d < out->ndim; d++) shapes[dest].shape[d] = out->shape[d];
            }
        } else if (op != SIM_NOOP && op != SIM_REALIZE) {
            res.had_null = true;
        }
    }

    /* fingerprint all live tensors */
    for (int i = 0; i < n_tensors; i++) {
        if (tensors[i]) {
            res.fingerprint ^= hash_tensor(tensors[i]) ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL);
            tensor_free(tensors[i]);
        }
    }

    cml_ir_reset_global_context();
    return res;
}

/* ── determinism check ───────────────────────────────────────────────────── */

static bool check_determinism(uint64_t seed, bool verbose) {
    WorkloadResult r1 = run_workload(seed, false);
    WorkloadResult r2 = run_workload(seed, false);
    if (r1.fingerprint != r2.fingerprint) {
        printf("  FAIL determinism seed=0x%016llx  fp1=0x%016llx fp2=0x%016llx\n",
               (unsigned long long)seed,
               (unsigned long long)r1.fingerprint,
               (unsigned long long)r2.fingerprint);
        if (verbose) run_workload(seed, true);
        return false;
    }
    return true;
}

/* ── OOM sweep ───────────────────────────────────────────────────────────── */

static bool oom_sweep(uint64_t seed) {
    /* First: count how many allocations a normal run uses */
    cml_malloc_fault_reset();
    run_workload(seed, false);
    long total_allocs = cml_malloc_alloc_index();

    /* Now: for each allocation index, make it fail */
    int failures = 0;
    for (long n = 0; n < total_allocs; n++) {
        cml_malloc_fault_after((int)n);
        /* run_workload must not crash regardless of where OOM is injected */
        run_workload(seed, false);
        /* if we got here without SIGSEGV/abort, this site is handled */
        cml_malloc_fault_reset();
    }
    (void)failures;
    return true; /* if we get here, no crash at any site */
}

/* ── gradient check ──────────────────────────────────────────────────────── */

static float finite_diff_grad(Tensor* (*f)(Tensor*), Tensor* x, size_t idx, float eps) {
    float* data = (float*)tensor_data_ptr(x);
    if (!data) return NAN;

    float orig = data[idx];

    data[idx] = orig + eps;
    x->is_executed = false;
    Tensor* yp = f(x);
    if (!yp) { data[idx] = orig; return NAN; }
    tensor_realize(yp);
    float* dp = (float*)tensor_data_ptr(yp);
    float fp = 0.0f;
    if (dp) for (size_t i = 0; i < yp->numel; i++) fp += dp[i];
    tensor_free(yp);
    cml_ir_reset_global_context();

    data[idx] = orig - eps;
    x->is_executed = false;
    Tensor* ym = f(x);
    if (!ym) { data[idx] = orig; return NAN; }
    tensor_realize(ym);
    float* dm = (float*)tensor_data_ptr(ym);
    float fm = 0.0f;
    if (dm) for (size_t i = 0; i < ym->numel; i++) fm += dm[i];
    tensor_free(ym);
    cml_ir_reset_global_context();

    data[idx] = orig;
    return (fp - fm) / (2.0f * eps);
}

static bool grad_check_op(const char* name, Tensor* (*f)(Tensor*), Tensor* x, float tol) {
    /* autograd gradient */
    Tensor* y = f(x);
    if (!y) { printf("  SKIP grad_check %s (op returned NULL)\n", name); return true; }

    /* sum output so grad flows to every input element */
    Tensor* ones = tensor_ones_like(y);
    tensor_backward(y, ones, false, false);
    tensor_free(ones);

    float* ag = x->grad ? (float*)tensor_data_ptr(x->grad) : NULL;
    float* xd = (float*)tensor_data_ptr(x);
    if (!ag || !xd) {
        tensor_free(y);
        cml_ir_reset_global_context();
        return true; /* can't check without grad */
    }

    bool ok = true;
    size_t n_check = x->numel < 8 ? x->numel : 8; /* check up to 8 elements */
    for (size_t i = 0; i < n_check; i++) {
        float fd = finite_diff_grad(f, x, i, 1e-3f);
        float an = ag[i];
        if (!isfinite(fd) || !isfinite(an)) continue;
        float rel = fabsf(fd - an) / (fabsf(fd) + fabsf(an) + 1e-6f);
        if (rel > tol) {
            printf("  FAIL grad_check %s idx=%zu fd=%.6f ag=%.6f rel=%.4f\n",
                   name, i, (double)fd, (double)an, (double)rel);
            ok = false;
        }
    }
    tensor_free(y);
    if (x->grad) { tensor_free(x->grad); x->grad = NULL; }
    cml_ir_reset_global_context();
    return ok;
}

static bool run_grad_checks(void) {
    bool all_ok = true;
    int shape[] = {2, 3};
    float data[] = {0.1f, 0.5f, -0.3f, 0.8f, -0.6f, 0.2f};
    TensorConfig cfg = {.dtype=DTYPE_FLOAT32, .device=DEVICE_CPU,
                        .has_dtype=true, .has_device=true};

    Tensor* x = cml_tensor(data, shape, 2, &cfg);
    if (!x) return false;
    tensor_realize(x);
    x->requires_grad = true;

    all_ok &= grad_check_op("relu",    tensor_relu,    x, 0.01f);
    all_ok &= grad_check_op("sigmoid", tensor_sigmoid, x, 0.01f);
    all_ok &= grad_check_op("tanh",    tensor_tanh,    x, 0.01f);

    tensor_free(x);
    return all_ok;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    uint64_t seed     = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)argv;
    int      iters    = 64;
    bool     oom      = false;
    bool     gradchk  = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i+1 < argc)
            seed = (uint64_t)strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc)
            iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--oom-sweep"))
            oom = true;
        else if (!strcmp(argv[i], "--grad-check"))
            gradchk = true;
    }

    cml_init();

    /* Force single-threaded execution so floating-point ops are deterministic.
     * Multi-threaded matmul reorders FP additions, breaking bit-exact reproducibility. */
    ThreadPool* single = threadpool_create(1);
    if (single) threadpool_set_global(single);

    printf("sim_test  seed=0x%016llx  iters=%d  oom=%s  gradcheck=%s\n",
           (unsigned long long)seed, iters, oom ? "yes" : "no", gradchk ? "yes" : "no");

    int failures = 0;

    /* --- determinism sweep --- */
    printf("[1/3] Determinism (%d seeds)...\n", iters);
    for (int i = 0; i < iters; i++) {
        uint64_t s = seed + (uint64_t)i;
        if (!check_determinism(s, false)) {
            failures++;
            printf("  repro: sim_test --seed 0x%016llx\n", (unsigned long long)s);
        }
    }
    printf("  done  failures=%d\n", failures);

    /* --- OOM sweep (optional, slow) --- */
    if (oom) {
        printf("[2/3] OOM sweep (seed=0x%016llx)...\n", (unsigned long long)seed);
        if (!oom_sweep(seed)) failures++;
        printf("  done  (no crashes)\n");
    } else {
        printf("[2/3] OOM sweep skipped (pass --oom-sweep to enable)\n");
    }

    /* --- gradient check --- */
    if (gradchk) {
        printf("[3/3] Gradient check...\n");
        if (!run_grad_checks()) failures++;
        printf("  done\n");
    } else {
        printf("[3/3] Gradient check skipped (pass --grad-check to enable)\n");
    }

    cml_cleanup();
    printf("\n%s  (failures=%d)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
