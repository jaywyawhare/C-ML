/* Structural validation of generated SPIR-V. A full validator isn't available
 * offline, but we can verify the two things the codegen bug broke:
 *   1. the module starts with the SPIR-V magic number,
 *   2. every instruction's word-count sums exactly to the module size
 *      (a malformed opcode/word-count desyncs this walk), and
 *   3. an OpExecutionMode (opcode 16) with the LocalSize(17) operand is present
 *      — previously opcode 17 was wrongly emitted as the opcode, so no valid
 *      OpExecutionMode existed and the instruction walk desynced. */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "cml.h"
#include "ops/ir/gpu/spirv_codegen.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"

#define SPIRV_MAGIC 0x07230203u
#define OP_EXECUTION_MODE 16u
#define EXEC_MODE_LOCAL_SIZE 17u

/* Returns 1 if the word stream is a well-formed SPIR-V module with a correct
 * LocalSize OpExecutionMode; 0 otherwise. */
static int validate_spirv(const uint32_t* w, size_t nbytes) {
    if (!w || nbytes < 5 * sizeof(uint32_t)) return 0;
    size_t n = nbytes / sizeof(uint32_t);
    if (w[0] != SPIRV_MAGIC) { printf("(bad magic 0x%08x) ", w[0]); return 0; }

    size_t i = 5;              /* skip the 5-word header */
    int found_exec_mode = 0;
    while (i < n) {
        uint32_t wc = w[i] >> 16;
        uint32_t op = w[i] & 0xFFFFu;
        if (wc == 0 || i + wc > n) { printf("(desync at word %zu wc=%u) ", i, wc); return 0; }
        if (op == OP_EXECUTION_MODE) {
            if (wc != 6 || w[i + 2] != EXEC_MODE_LOCAL_SIZE) { printf("(bad ExecutionMode) "); return 0; }
            found_exec_mode = 1;
        }
        i += wc;
    }
    if (i != n) { printf("(instructions don't fill module) "); return 0; }
    if (!found_exec_mode) { printf("(no OpExecutionMode) "); return 0; }
    return 1;
}

#undef TEST
#define TEST(label, cond) do { \
    tests_run++; printf("  %-40s ", label); \
    if (cond) { tests_passed++; printf("[PASS]\n"); } else printf("[FAIL]\n"); \
} while (0)

int main(void) {
    printf("SPIR-V codegen validation:\n");
    CMLSPIRVCodegen* cg = cml_spirv_codegen_create();
    if (!cg) { printf("codegen_create failed\n"); return 1; }

    size_t sz = 0;
    uint32_t* w;

    w = cml_spirv_gen_unary(cg, UOP_NEG, "neg", &sz);
    TEST("unary (neg)", validate_spirv(w, sz));
    if (w) cml_free(w);

    w = cml_spirv_gen_binary(cg, UOP_ADD, "add", &sz);
    TEST("binary (add)", validate_spirv(w, sz));
    if (w) cml_free(w);

    w = cml_spirv_gen_fill(cg, 1.5f, "fill", &sz);
    TEST("fill", validate_spirv(w, sz));
    if (w) cml_free(w);

    w = cml_spirv_gen_matmul(cg, "matmul", &sz);
    TEST("matmul", validate_spirv(w, sz));
    if (w) cml_free(w);

    cml_spirv_codegen_destroy(cg);
    return TEST_SUMMARY();
}
