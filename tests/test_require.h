/**
 * An assertion that survives NDEBUG.
 *
 * `assert()` compiles to nothing under NDEBUG, which the default Release build
 * defines. That silently disabled 474 assertions across 18 test files: they
 * verified nothing and passed by merely running to completion. Worse, tests that
 * wrapped a side-effecting call in the assert --
 *
 *     assert(torch_selective_build_from_string("add,mul", &cfg) == 0);
 *
 * -- lost the call itself, so the code under test never ran and the next line
 * read an uninitialised struct.
 *
 * REQUIRE evaluates its argument exactly once in every build type and aborts on
 * failure, so a test means the same thing regardless of how it was compiled.
 * Prefer it to assert() in tests; the build also passes -UNDEBUG, but that
 * protects only this project's own build files.
 */
#ifndef CML_TEST_REQUIRE_H
#define CML_TEST_REQUIRE_H

#include <stdio.h>
#include <stdlib.h>

#define REQUIRE(expr)                                                                              \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            fprintf(stderr, "%s:%d: REQUIRE failed: %s\n", __FILE__, __LINE__, #expr);             \
            fflush(stderr);                                                                        \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

#endif /* CML_TEST_REQUIRE_H */
