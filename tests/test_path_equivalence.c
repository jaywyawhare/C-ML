/* The same graph must produce the same numbers on every execution path.
 *
 * C-ML computes a given graph several different ways depending on runtime
 * flags: the per-node LLVM JIT, the CPU interpreter, the fused-elementwise
 * kernel, Winograd or direct convolution, and so on. Each is a separate
 * implementation of the same semantics, and they drifted apart silently -- the
 * JIT indexed broadcast operands with `i % numel` while the interpreter used a
 * shape-aware map, so `[2,3] op [2,1]` (the shape at the centre of layernorm
 * and softmax) mixed rows together only when the JIT was enabled, which is the
 * default. Both paths individually "worked"; only comparing them exposed it.
 *
 * This test runs one battery under each configuration and requires identical
 * output. Flags are read once per process, so it re-executes itself rather than
 * trying to switch them in-process.
 *
 * Any mismatch is a real bug in one of the paths -- the battery is built from
 * exactly representable values so that legitimate float reassociation cannot
 * account for a difference.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "cml.h"
#include "test_require.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* Small integers, exactly representable, so any difference is a logic bug. */
static Tensor* mk(const int* shape, int ndim, int seed) {
    TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                      .has_dtype = true, .has_device = true};
    Tensor* t = tensor_zeros((int*)shape, ndim, &c);
    for (size_t i = 0; i < t->numel; i++)
        tensor_set_float(t, i, (float)(((int)i * 7 + seed * 13) % 17) - 8.0f);
    return t;
}

static void emit(const char* name, Tensor* r) {
    if (!r) { printf("%-22s NULL\n", name); return; }
    tensor_ensure_executed(r);
    double sum = 0, asum = 0;
    for (size_t i = 0; i < r->numel; i++) {
        double v = (double)tensor_get_float(r, i);
        sum += v;
        asum += (v < 0 ? -v : v);
    }
    printf("%-22s n=%-7zu sum=%.6f asum=%.6f\n", name, r->numel, sum, asum);
    cml_reset_ir_context();
}

static void run_battery(void) {
    const int m[2]   = {4, 6};
    const int row[1] = {6};
    const int col[2] = {4, 1};
    const int sq[2]  = {5, 5};
    /* large enough to cross the SIMD and threadpool thresholds */
    const int big[2]  = {256, 128};
    const int bigc[2] = {256, 1};

    emit("add",           uop_add(mk(m, 2, 1), mk(m, 2, 2)));
    emit("add row-bcast", uop_add(mk(m, 2, 1), mk(row, 1, 3)));
    emit("add col-bcast", uop_add(mk(m, 2, 1), mk(col, 2, 4)));
    emit("mul col-bcast", uop_mul(mk(m, 2, 1), mk(col, 2, 4)));
    emit("chain",         uop_relu(uop_add(uop_mul(mk(m, 2, 1), mk(m, 2, 2)), mk(m, 2, 3))));
    emit("relu",          uop_relu(mk(m, 2, 5)));
    emit("neg abs",       uop_abs(uop_neg(mk(m, 2, 1))));
    emit("sum all",       uop_sum(mk(m, 2, 1), NULL));
    emit("max all",       uop_max_reduce(mk(m, 2, 1), NULL));
    {
        static int d[1] = {1};
        ReduceParams rp = {.dims = d, .num_dims = 1, .keepdim = false};
        emit("sum dim1", uop_sum(mk(m, 2, 1), &rp));
        emit("max dim1", uop_max_reduce(mk(m, 2, 1), &rp));
    }
    emit("matmul",        uop_matmul(mk(sq, 2, 1), mk(sq, 2, 2)));
    emit("matmul relu",   uop_relu(uop_matmul(mk(sq, 2, 1), mk(sq, 2, 2))));
    emit("cumsum",        uop_cumsum(mk(m, 2, 1), 1));
    emit("sort",          uop_sort(mk(m, 2, 1), 1, false));
    emit("big add",       uop_add(mk(big, 2, 1), mk(big, 2, 2)));
    emit("big col-bcast", uop_add(mk(big, 2, 1), mk(bigc, 2, 4)));
    emit("big relu chain",uop_relu(uop_mul(mk(big, 2, 1), mk(big, 2, 2))));
    emit("big sum",       uop_sum(mk(big, 2, 1), NULL));
    {
        const int x[4] = {2, 3, 8, 8}, w[4] = {4, 3, 3, 3};
        static int ks[2] = {3, 3}, st[2] = {1, 1}, pd[2] = {1, 1}, dl[2] = {1, 1};
        Conv2DParams p = {.kernel_size = ks, .stride = st, .padding = pd,
                          .dilation = dl, .groups = 1, .bias = false};
        emit("conv2d", uop_conv2d(mk(x, 4, 1), mk(w, 4, 2), NULL, &p));
    }
    {
        const int x[4] = {2, 3, 8, 8};
        Pool2DParams p = {.kernel_size = {2, 2}, .stride = {2, 2}, .padding = {0, 0},
                          .dilation = {1, 1}, .count_include_pad = false};
        emit("maxpool2d", uop_maxpool2d(mk(x, 4, 1), &p));
        emit("avgpool2d", uop_avgpool2d(mk(x, 4, 1), &p));
    }
}

/* Configurations that select genuinely different implementations. */
static const char* CONFIGS[] = {
    "JIT=0", "DISABLE_JIT=1", "JIT=2",
    "NOOPT=1", "DISABLE_FUSION=1", "NO_MEMORY_PLANNER=1",
    "SPLIT_REDUCEOP=0", "WINO=0", "WINO=1", "CACHELEVEL=0",
};
#define NUM_CONFIGS ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))

/* Re-exec self with `cfg` set, capturing stdout. */
static char* capture(const char* cfg) {
    int fds[2];
    if (pipe(fds) != 0) return NULL;
    pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return NULL; }
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        close(fds[1]);
        if (cfg) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s", cfg);
            char* eq = strchr(buf, '=');
            if (eq) { *eq = '\0'; setenv(buf, eq + 1, 1); }
        }
        execl("/proc/self/exe", "test_path_equivalence", "--battery", (char*)NULL);
        _exit(127);
    }
    close(fds[1]);
    size_t cap = 1 << 16, len = 0;
    char* out = malloc(cap);
    ssize_t n;
    while (out && (n = read(fds[0], out + len, cap - len - 1)) > 0) {
        len += (size_t)n;
        if (len + 1 >= cap) { cap *= 2; char* t = realloc(out, cap); if (!t) break; out = t; }
    }
    if (out) out[len] = '\0';
    close(fds[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    return out;
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--battery") == 0) {
        cml_init();
        run_battery();
        cml_cleanup();
        return 0;
    }

    printf("Execution-path equivalence (%d configurations):\n", NUM_CONFIGS);
    char* base = capture(NULL);
    REQUIRE(base != NULL && base[0] != '\0');

    int failures = 0;
    for (int i = 0; i < NUM_CONFIGS; i++) {
        char* got = capture(CONFIGS[i]);
        if (!got || got[0] == '\0') {
            printf("  %-20s produced no output\n", CONFIGS[i]);
            failures++;
        } else if (strcmp(base, got) != 0) {
            printf("  %-20s DIFFERS from the default path\n", CONFIGS[i]);
            /* first differing line, for diagnosis */
            const char *a = base, *b = got;
            while (*a && *b) {
                const char* ae = strchr(a, '\n');
                const char* be = strchr(b, '\n');
                size_t al = ae ? (size_t)(ae - a) : strlen(a);
                size_t bl = be ? (size_t)(be - b) : strlen(b);
                if (al != bl || strncmp(a, b, al) != 0) {
                    printf("      default: %.*s\n", (int)al, a);
                    printf("      %-8s %.*s\n", CONFIGS[i], (int)bl, b);
                    break;
                }
                if (!ae || !be) break;
                a = ae + 1;
                b = be + 1;
            }
            failures++;
        } else {
            printf("  %-20s identical\n", CONFIGS[i]);
        }
        free(got);
    }
    free(base);

    printf("\n%d configurations, %d mismatched\n", NUM_CONFIGS, failures);
    return failures ? 1 : 0;
}
