#ifndef CML_CORE_FLAGS_H
#define CML_CORE_FLAGS_H

#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Central runtime-flag registry (inspired by tinygrad's ContextVar).
 *
 * Every flag is read once from the environment on first access (or explicitly
 * via cml_flags_init()) and is thereafter queryable with cml_flag(). This
 * replaces the scattered getenv() calls across the codebase with a single
 * documented table that DEBUG>=1 can dump.
 *
 * Values may be overridden programmatically for a scope with
 * cml_flag_push()/cml_flag_pop(), mirroring tinygrad's `with Context(FLAG=x):`.
 *
 * Each entry in the list is: ID, ENV-VAR NAME, DEFAULT VALUE, DESCRIPTION.
 */
#define CML_FLAG_LIST(X)                                                                            \
    /* Tier 1 -- leveled debug + core toggles */                                                   \
    X(DEBUG, "DEBUG", 0,                                                                            \
      "Leveled debug output (1-7): 1 devices, 2 timings, 3 opts, 4 codegen, 5 IR, 6 linear IR, "   \
      "7 asm")                                                                                      \
    X(NO_COLOR, "NO_COLOR", 0, "Disable ANSI color in log output")                                  \
    X(NOOPT, "NOOPT", 0, "Disable IR/kernel optimization passes")                                   \
    X(PROFILE, "PROFILE", 0, "Enable kernel/op profiling")                                          \
    X(NO_MEMORY_PLANNER, "NO_MEMORY_PLANNER", 0,                                                    \
      "Disable the graph memory planner (allocate each buffer independently)")                      \
    X(WINO, "WINO", -1, "Winograd conv override: -1 auto, 0 force off, 1 force on")                 \
    X(CHECK_OOB, "CHECK_OOB", 0, "Enable out-of-bounds index checking in kernels")                  \
    X(VALIDATE_WITH_CPU, "VALIDATE_WITH_CPU", 0,                                                    \
      "Re-run graphs on CPU and diff results against the active backend")                           \
    X(CACHELEVEL, "CACHELEVEL", 2, "Kernel cache level: 0 none, 1 memory, 2 memory+disk")           \
    /* Tier 2 -- more work / backend dependent */                                                  \
    X(JIT, "JIT", 1, "JIT control: 0 off, 1 on, 2 on but graphs disabled")                          \
    X(TC, "TC", 1, "Use tensor cores when available")                                               \
    X(TC_SELECT, "TC_SELECT", -1, "Select a specific tensor-core config (-1 auto)")                 \
    X(TC_OPT, "TC_OPT", 0, "Tensor-core optimization aggressiveness (0-2)")                         \
    X(TRANSCENDENTAL, "TRANSCENDENTAL", 1,                                                          \
      "Transcendental ops: 0 hardware, 1 auto, 2 always polynomial approximation")                 \
    X(NOLOCALS, "NOLOCALS", 0, "Disable use of local/shared memory in kernels")                     \
    X(SPLIT_REDUCEOP, "SPLIT_REDUCEOP", 1, "Split large reduce ops for parallelism")                \
    X(IGNORE_BEAM_CACHE, "IGNORE_BEAM_CACHE", 0, "Ignore the on-disk BEAM search cache")            \
    X(FUSE_OPTIM, "FUSE_OPTIM", 0, "Fuse the optimizer update into the backward graph")             \
    X(MAX_BUFFER_SIZE, "MAX_BUFFER_SIZE", 0, "Cap single buffer allocation size in bytes (0 = "     \
                                             "unlimited)")                                          \
    /* Pre-existing flags, centralized here */                                                     \
    X(BEAM, "BEAM", 0, "Number of beams in kernel beam search (0 = disabled)")                      \
    X(DISABLE_FUSION, "DISABLE_FUSION", 0, "Disable operator fusion in the scheduler")              \
    X(DISABLE_JIT, "DISABLE_JIT", 0, "Disable JIT compilation (force interpreter/BLAS path)")       \
    X(VIZ, "VIZ", 0, "Launch the graph/kernel visualizer")

typedef enum {
#define CML_FLAG_ENUM(id, name, dflt, desc) CML_FLAG_##id,
    CML_FLAG_LIST(CML_FLAG_ENUM)
#undef CML_FLAG_ENUM
        CML_FLAG_COUNT
} CmlFlag;

/* Read all flags from the environment. Idempotent and thread-safe; called
 * automatically on first cml_flag() access, and explicitly from cml_init(). */
void cml_flags_init(void);

/* Current integer value of a flag. */
int cml_flag(CmlFlag id);

/* Convenience: true when the flag is non-zero. */
bool cml_flag_enabled(CmlFlag id);

/* True if the flag was explicitly set via the environment (vs. left at default).
 * Useful for tri-state overrides such as WINO. */
bool cml_flag_was_set(CmlFlag id);

/* The environment-variable name / description for a flag. */
const char* cml_flag_name(CmlFlag id);
const char* cml_flag_desc(CmlFlag id);

/* Scoped override, mirroring tinygrad `with Context(FLAG=value):`.
 * cml_flag_push() returns the previous value; pass it back to cml_flag_pop(). */
int cml_flag_push(CmlFlag id, int value);
void cml_flag_pop(CmlFlag id, int previous);

/* Print every flag whose value differs from its default (used by DEBUG>=1). */
void cml_flags_dump(FILE* out);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_FLAGS_H
