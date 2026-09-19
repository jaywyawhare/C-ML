/* VIZ and NO_EXPORT ask for opposite things, so the combination is rejected at
 * init rather than resolved by precedence -- a run started with VIZ=1 must never
 * silently produce nothing. Each flag on its own must still initialise. */
#include <stdio.h>
#include <stdlib.h>
#include "cml.h"
#include "test_harness.h"

static const char* program_path(void);

/* cml_init() latches process-global state, so each combination is exercised in
 * a fresh child rather than by re-initialising in place. */
static int init_rc_with(const char* viz, const char* no_export) {
    /* VIZ_LAUNCHED=1 in the child: VIZ=1 otherwise forks the dashboard launcher,
     * which binds a port and never returns, so the probe hangs instead of
     * reporting an exit code. This test is about whether cml_init() accepts the
     * flag combination, not about the launcher.
     *
     * It passed before only by accident -- a dashboard already running on the
     * port made the launcher exit immediately. With nothing on the port it
     * starts its own server and blocks. */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "VIZ_LAUNCHED=1 %s %s %s --probe-init", viz ? viz : "",
             no_export ? no_export : "", program_path());
    return system(cmd);
}

static const char* g_argv0 = NULL;
static const char* program_path(void) { return g_argv0; }

int main(int argc, char** argv) {
    g_argv0 = argv[0];

    /* Child mode: report whether cml_init() accepted this environment. */
    if (argc > 1 && argv[1] && argv[1][0] == '-') {
        return cml_init() == 0 ? 0 : 3;
    }

    printf("=== VIZ / NO_EXPORT mutual exclusion ===\n");

    CHECK("neither flag: init succeeds", init_rc_with(NULL, NULL) == 0);
    CHECK("VIZ alone: init succeeds", init_rc_with("VIZ=1", NULL) == 0);
    CHECK("NO_EXPORT alone: init succeeds", init_rc_with(NULL, "NO_EXPORT=1") == 0);

    int both = init_rc_with("VIZ=1", "NO_EXPORT=1");
    CHECK("VIZ + NO_EXPORT: init is rejected", both != 0);

    /* VIZ=0 is not "VIZ requested", so it must not trip the conflict. */
    CHECK("VIZ=0 with NO_EXPORT: init succeeds", init_rc_with("VIZ=0", "NO_EXPORT=1") == 0);

    return TEST_SUMMARY();
}
