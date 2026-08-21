/**
 * Tally-and-report scaffolding for the test suites.
 *
 * Each test file is a separate program, so the counters are file-scope
 * definitions rather than externs.
 */
#ifndef CML_TEST_HARNESS_H
#define CML_TEST_HARNESS_H

#include <stdio.h>

static int tests_run    = 0;
static int tests_passed = 0;

/* Call test_<name>(), which returns non-zero on success, and record it.
 *
 * A suite needing per-case setup or teardown redefines TEST in terms of
 * TEST_CASE rather than reimplementing the tally. */
#define TEST_CASE(name)                                                                            \
    do {                                                                                           \
        tests_run++;                                                                               \
        printf("  %-55s ", #name);                                                                 \
        fflush(stdout); /* so a crashing test is attributed to the right name */                   \
        if (test_##name()) {                                                                       \
            tests_passed++;                                                                        \
            printf("[PASS]\n");                                                                    \
        } else {                                                                                   \
            printf("[FAIL]\n");                                                                    \
        }                                                                                          \
    } while (0)

#define TEST(name) TEST_CASE(name)

/* Inline-condition variant for suites written as straight-line code rather
 * than test_ functions: CHECK("label", cond) tallies like TEST_CASE. */
#define CHECK(name, cond)                                                                          \
    do {                                                                                           \
        tests_run++;                                                                               \
        printf("  %-55s ", name);                                                                  \
        if (cond) {                                                                                \
            tests_passed++;                                                                        \
            printf("[PASS]\n");                                                                    \
        } else {                                                                                   \
            printf("[FAIL]\n");                                                                    \
        }                                                                                          \
    } while (0)

/* Print the tally and yield main()'s return value: 0 only if everything passed.
 * Usage: `return TEST_SUMMARY();` -- each suite already prints its own title on
 * the way in, so this only reports the count. */
#define TEST_SUMMARY()                                                                             \
    (printf("\nResults: %d/%d passed\n", tests_passed, tests_run),                                 \
     tests_passed == tests_run ? 0 : 1)

#endif /* CML_TEST_HARNESS_H */
