/* NOOPT=1 must actually disable fusion, and disabling fusion must not change
 * results.
 *
 * Both halves had failed silently. The fusers were reached from three call
 * sites and only one consulted NOOPT, so "optimization disabled" runs still
 * executed fused kernels -- which made NOOPT useless as the reference mode for
 * differential testing, and hid a JIT fused-kernel bug that returned all zeros
 * for any chain ending in a ReLU. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cml.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/execution.h"
#include "test_harness.h"

/* Flags latch on first use, so each configuration is measured in a fresh child
 * that reports its fused-node count and result on stdout. */
static const char* g_argv0 = NULL;

typedef struct {
    int enabled;
    int fused;
    float v[4];
} Probe;

static Tensor* mk(int r, int c, const float* vals) {
    Tensor* t = cml_zeros_2d(r, c);
    float* d  = (float*)tensor_data_ptr(t);
    for (int i = 0; i < r * c; i++)
        d[i] = vals[i];
    return t;
}

/* add -> relu: the smallest chain the fuser collapses, and the one whose fused
 * kernel was miscompiled. */
static int run_probe(void) {
    cml_init();
    float x[4] = {58, 64, 139, 154};
    float y[4] = {2, -100, 1, -200};

    Tensor* out  = cml_relu(cml_add(mk(2, 2, x), mk(2, 2, y)));
    CMLGraph_t g = tensor_get_ir_context(out);
    tensor_ensure_executed(out);

    int fused = 0;
    for (struct IRNode* n = g ? g->head : NULL; n; n = n->next)
        if (n->type == UOP_FUSED_ELEMENTWISE)
            fused++;

    float* d = (float*)tensor_data_ptr(out);
    printf("PROBE %d %d %.4f %.4f %.4f %.4f\n", cml_ir_fusion_enabled(), fused, d[0], d[1], d[2],
           d[3]);
    fflush(stdout);
    cml_reset_ir_context();
    return 0;
}

/* Every flag this test reasons about is pinned explicitly, never left to the
 * ambient environment: the probes are children, so running the suite itself
 * under NOOPT=1 would otherwise silently turn the baseline into a second NOOPT
 * run. `on` names the one flag being switched to its non-default value. */
static int probe_with(const char* on, Probe* p) {
    const char* noopt  = "0";
    const char* nofuse = "0";
    const char* sched  = "1";
    if (on && strcmp(on, "NOOPT") == 0)
        noopt = "1";
    else if (on && strcmp(on, "DISABLE_FUSION") == 0)
        nofuse = "1";
    else if (on && strcmp(on, "FUSION_SCHEDULER") == 0)
        sched = "0";

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "NOOPT=%s DISABLE_FUSION=%s FUSION_SCHEDULER=%s %s --probe", noopt,
             nofuse, sched, g_argv0);
    FILE* f = popen(cmd, "r");
    if (!f)
        return 0;
    char line[256];
    int got = 0;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "PROBE ", 6) == 0)
            got = sscanf(line + 6, "%d %d %f %f %f %f", &p->enabled, &p->fused, &p->v[0], &p->v[1],
                         &p->v[2], &p->v[3]) == 6;
    pclose(f);
    return got;
}

static int values_match(const Probe* a, const Probe* b) {
    for (int i = 0; i < 4; i++) {
        float d = a->v[i] - b->v[i];
        if (d < 0)
            d = -d;
        if (d > 1e-4f)
            return 0;
    }
    return 1;
}

static Probe g_on, g_noopt, g_nofuse, g_nosched;

static int test_fusion_on_by_default(void) { return g_on.enabled == 1 && g_on.fused >= 1; }

/* The point of the fix: NOOPT is documented as disabling optimization passes,
 * and fusion is one. */
static int test_noopt_disables_fusion(void) { return g_noopt.enabled == 0 && g_noopt.fused == 0; }

static int test_disable_fusion_disables_fusion(void) {
    return g_nofuse.enabled == 0 && g_nofuse.fused == 0;
}

static int test_fusion_scheduler_zero_disables_fusion(void) {
    return g_nosched.enabled == 0 && g_nosched.fused == 0;
}

/* Fusion is a performance transform; it must not move the numbers. This is the
 * assertion that catches a miscompiled fused kernel. */
static int test_fusion_does_not_change_results(void) {
    return values_match(&g_on, &g_noopt) && values_match(&g_on, &g_nofuse) &&
           values_match(&g_on, &g_nosched);
}

static int test_result_is_correct(void) {
    Probe want = {0, 0, {60.0f, 0.0f, 140.0f, 0.0f}};
    return values_match(&g_on, &want);
}

int main(int argc, char** argv) {
    g_argv0 = argv[0];
    if (argc > 1 && strcmp(argv[1], "--probe") == 0)
        return run_probe();

    printf("=== NOOPT / fusion gating ===\n");
    if (!probe_with(NULL, &g_on) || !probe_with("NOOPT", &g_noopt) ||
        !probe_with("DISABLE_FUSION", &g_nofuse) || !probe_with("FUSION_SCHEDULER", &g_nosched)) {
        printf("  probe subprocess failed\n");
        return 1;
    }

    TEST(fusion_on_by_default);
    TEST(noopt_disables_fusion);
    TEST(disable_fusion_disables_fusion);
    TEST(fusion_scheduler_zero_disables_fusion);
    TEST(fusion_does_not_change_results);
    TEST(result_is_correct);

    return TEST_SUMMARY();
}
