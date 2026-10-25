/* VIZ and NO_EXPORT ask for opposite things, so the combination is rejected at
 * init rather than resolved by precedence -- a run started with VIZ=1 must never
 * silently produce nothing. Each flag on its own must still initialise. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cml.h"
#include "ops/ir/flamegraph.h"
#include "ops/ir/rewrite_trace.h"
#include "test_harness.h"

static const char* program_path(void);

/**
 * cml_init() latches process-global state, so each combination is exercised in
 * a fresh child rather than by re-initialising in place.
 * @param viz (const char*)
 * @param no_export (const char*)
 * @param probe (const char*)
 * @return (int)
 */
static int init_rc_with(const char* viz, const char* no_export, const char* probe) {
    /* VIZ_LAUNCHED=1 in the child: VIZ=1 otherwise forks the dashboard launcher,
     * which binds a port and never returns, so the probe hangs instead of
     * reporting an exit code. This test is about whether cml_init() accepts the
     * flag combination, not about the launcher.
     *
     * It passed before only by accident -- a dashboard already running on the
     * port made the launcher exit immediately. With nothing on the port it
     * starts its own server and blocks. */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "VIZ_LAUNCHED=1 %s %s %s %s", viz ? viz : "",
             no_export ? no_export : "", program_path(), probe);
    return system(cmd);
}

static const char* g_argv0 = NULL;
static const char* program_path(void) { return g_argv0; }

int main(int argc, char** argv) {
    g_argv0 = argv[0];

    /* Child mode: report whether cml_init() accepted this environment and, for --probe-gates,
     * whether it left any VIZ-only path on. */
    if (argc > 1 && argv[1] && argv[1][0] == '-') {
        if (cml_init() != 0)
            return 3;
        if (strcmp(argv[1], "--probe-gates") == 0)
            return cml_viz_enabled() || cml_flame_enabled() || cml_rewrite_trace_enabled() ? 4 : 0;
        return 0;
    }

    /* Each probe must see exactly the flags it passes, not ones this test inherited. */
    unsetenv("VIZ");
    unsetenv("NO_EXPORT");

    printf("=== VIZ / NO_EXPORT mutual exclusion ===\n");

    CHECK("neither flag: init succeeds", init_rc_with(NULL, NULL, "--probe-init") == 0);
    CHECK("VIZ alone: init succeeds", init_rc_with("VIZ=1", NULL, "--probe-init") == 0);
    CHECK("NO_EXPORT alone: init succeeds", init_rc_with(NULL, "NO_EXPORT=1", "--probe-init") == 0);

    int both = init_rc_with("VIZ=1", "NO_EXPORT=1", "--probe-init");
    CHECK("VIZ + NO_EXPORT: init is rejected", both != 0);

    /* VIZ=0 is not "VIZ requested", so it must not trip the conflict. */
    CHECK("VIZ=0 with NO_EXPORT: init succeeds",
          init_rc_with("VIZ=0", "NO_EXPORT=1", "--probe-init") == 0);

    /* Every VIZ-only path reads the one flag: values that mean off turn all of them off. */
    CHECK("VIZ=false leaves every VIZ path off",
          init_rc_with("VIZ=false", NULL, "--probe-gates") == 0);
    CHECK("an empty VIZ leaves every VIZ path off",
          init_rc_with("VIZ=", NULL, "--probe-gates") == 0);
    CHECK("VIZ=0 leaves every VIZ path off", init_rc_with("VIZ=0", NULL, "--probe-gates") == 0);
    CHECK("VIZ=true turns the VIZ paths on", init_rc_with("VIZ=true", NULL, "--probe-gates") != 0);

    return TEST_SUMMARY();
}
