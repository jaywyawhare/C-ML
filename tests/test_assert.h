/**
 * The bail-on-first-failure idiom, used by the codegen suites.
 *
 * Their cases are `void` functions that abandon the case the moment an
 * assertion fails, rather than returning a pass flag -- so they count failures
 * rather than passes, and need scaffolding distinct from test_harness.h. Do not
 * include both in one file: they define different counters for different
 * bookkeeping.
 */
#ifndef CML_TEST_ASSERT_H
#define CML_TEST_ASSERT_H

#include <stdio.h>

static int tests_passed = 0;
static int tests_failed = 0;

/* Abandon the current case, recording the failure. Valid only inside a void
 * test function -- the bare `return` is the whole point of the idiom. */
#define ASSERT(cond, msg)                                                                          \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__);                                       \
            tests_failed++;                                                                        \
            return;                                                                                \
        }                                                                                          \
    } while (0)

#endif /* CML_TEST_ASSERT_H */
