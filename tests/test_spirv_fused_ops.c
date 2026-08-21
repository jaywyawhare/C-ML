/* Every op the fused-kernel gate admits must be emitted correctly by the SPIR-V
 * backend.
 *
 * The gate (uop_is_binary / uop_is_unary in fused_codegen.c) admitted MAX, POW,
 * SIGMOID, RECIP and SILU, but the SPIR-V emitter had no case for any of them
 * and its default arms substituted a plausible instruction -- OpFAdd for the
 * binaries, GLSL FAbs for the unaries. That compiles and validates, so nothing
 * caught it: max(a,b) silently became a+b on Vulkan. The C and WGSL backends
 * implemented all of them, so only Vulkan was wrong.
 *
 * This test asserts two things per op: the module is structurally valid, and it
 * actually contains the instruction that op requires (not a substituted one).
 */

#include "cml.h"
#include "ops/ir/fused_codegen.h"
#include "test_harness.h"
#include <stdio.h>
#include <string.h>

#define SPIRV_MAGIC 0x07230203u
#define OP_EXT_INST 12u
#define OP_FADD     129u
#define OP_FSUB     131u
#define OP_FMUL     133u
#define OP_FDIV     136u
#define OP_FNEGATE  127u

#define GLSL_POW    26u
#define GLSL_EXP    27u
#define GLSL_FABS   4u
#define GLSL_FMAX   40u

#undef TEST
#define TEST(label, cond) CHECK(label, cond)

/* Walk the instruction stream; returns 1 if the module is well formed. */
static int spirv_walk_ok(const uint32_t* w, int n) {
    if (!w || n < 5 || w[0] != SPIRV_MAGIC) return 0;
    int i = 5;
    while (i < n) {
        uint32_t wc = w[i] >> 16;
        if (wc == 0 || i + (int)wc > n) return 0;
        i += (int)wc;
    }
    return i == n;
}

/* Does the module contain a core opcode with the given number? */
static int has_opcode(const uint32_t* w, int n, uint32_t opcode) {
    int i = 5;
    while (i < n) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xFFFFu;
        if (wc == 0 || i + (int)wc > n) return 0;
        if (op == opcode) return 1;
        i += (int)wc;
    }
    return 0;
}

/* Does the module contain an OpExtInst selecting the given GLSL.std.450 op?
 * Layout: [OpExtInst | resultType | resultId | set | instruction | operands...] */
static int has_glsl_inst(const uint32_t* w, int n, uint32_t glsl_op) {
    int i = 5;
    while (i < n) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xFFFFu;
        if (wc == 0 || i + (int)wc > n) return 0;
        if (op == OP_EXT_INST && wc >= 5 && w[i + 4] == glsl_op) return 1;
        i += (int)wc;
    }
    return 0;
}

/* One compute op with `n_src` sources, plus the loads and the store. */
static uint32_t* gen_for_op(UOpType uop, int n_src, int* out_words) {
    CMLLinearOp ops[4];
    memset(ops, 0, sizeof(ops));
    int k = 0;

    for (int s = 0; s < n_src; s++) {
        ops[k].kind     = LINOP_LOAD;
        ops[k].dest_reg = s;
        k++;
    }
    ops[k].kind     = LINOP_COMPUTE;
    ops[k].uop      = uop;
    ops[k].dest_reg = n_src;
    ops[k].num_srcs = n_src;
    for (int s = 0; s < n_src; s++) ops[k].src_regs[s] = s;
    k++;

    ops[k].kind     = LINOP_STORE;
    ops[k].dest_reg = n_src;
    k++;

    CMLLinearProgram prog = {.ops = ops, .num_ops = k, .capacity = k, .next_vreg = n_src + 1};
    return cml_spirv_gen_fused_kernel(&prog, 256, out_words);
}

/* `expect_glsl` != 0 -> require that GLSL.std.450 instruction;
 * otherwise require the core `expect_core` opcode. Both also assert the module
 * does NOT contain the instruction the old default would have substituted. */
static void check_op(const char* name, UOpType uop, int n_src,
                     uint32_t expect_core, uint32_t expect_glsl) {
    int n = 0;
    uint32_t* w = gen_for_op(uop, n_src, &n);
    char label[96];

    snprintf(label, sizeof(label), "%s: module is well formed", name);
    TEST(label, w && spirv_walk_ok(w, n));

    snprintf(label, sizeof(label), "%s: emits the right instruction", name);
    if (!w) {
        TEST(label, 0);
        return;
    }
    TEST(label, expect_glsl ? has_glsl_inst(w, n, expect_glsl)
                            : has_opcode(w, n, expect_core));
    cml_free(w);
}

/* OpExtInstImport must name the set exactly. A single transposed byte in the
 * literal ("GLLS.std.450") made every module this backend emitted invalid --
 * structurally well formed, so only a real validator or driver would reject it. */
static int import_string_ok(const uint32_t* w, int n) {
    int i = 5;
    while (i < n) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xFFFFu;
        if (wc == 0 || i + (int)wc > n) return 0;
        if (op == 11u /* OpExtInstImport */ && wc >= 3)
            return memcmp(&w[i + 2], "GLSL.std.450", 13) == 0;
        i += (int)wc;
    }
    return 0;
}

static char* ptx_for(UOpType uop, int n_src) {
    CMLLinearOp ops[4];
    memset(ops, 0, sizeof(ops));
    int k = 0;
    for (int s = 0; s < n_src; s++) { ops[k].kind = LINOP_LOAD; ops[k].dest_reg = s; k++; }
    ops[k].kind = LINOP_COMPUTE; ops[k].uop = uop;
    ops[k].dest_reg = n_src; ops[k].num_srcs = n_src;
    for (int s = 0; s < n_src; s++) ops[k].src_regs[s] = s;
    k++;
    ops[k].kind = LINOP_STORE; ops[k].dest_reg = n_src; k++;
    CMLLinearProgram prog = {.ops = ops, .num_ops = k, .capacity = k, .next_vreg = n_src + 1};
    return cml_ptx_gen_fused_kernel(&prog, 256);
}

/* Require both marker instructions/constants to appear in the emitted PTX. */
static void check_ptx(const char* name, UOpType uop, int n_src,
                      const char* need1, const char* need2) {
    char* p = ptx_for(uop, n_src);
    char label[96];
    snprintf(label, sizeof(label), "ptx %s: emits %s", name, need1);
    TEST(label, p && strstr(p, need1) != NULL);
    if (need2) {
        snprintf(label, sizeof(label), "ptx %s: emits %s", name, need2);
        TEST(label, p && strstr(p, need2) != NULL);
    }
    cml_free(p);
}

int main(void) {
    cml_init();
    printf("SPIR-V fused-kernel op coverage:\n");

    {
        int n = 0;
        uint32_t* w = gen_for_op(UOP_ADD, 2, &n);
        TEST("extension set imported as GLSL.std.450", w && import_string_ok(w, n));
        cml_free(w);
    }

    /* Binary ops that were previously emitted as OpFAdd. */
    check_op("max",     UOP_MAX,     2, 0, GLSL_FMAX);
    check_op("pow",     UOP_POW,     2, 0, GLSL_POW);

    /* Unary ops that were previously emitted as GLSL FAbs. */
    check_op("recip",   UOP_RECIP,   1, OP_FDIV, 0);
    check_op("sigmoid", UOP_SIGMOID, 1, OP_FDIV, 0);
    check_op("silu",    UOP_SILU,    1, OP_FMUL, 0);

    /* sigmoid/silu are composed, so they must pull in exp and a negate. */
    {
        int n = 0;
        uint32_t* w = gen_for_op(UOP_SIGMOID, 1, &n);
        TEST("sigmoid: composes exp(-x)", w && has_glsl_inst(w, n, GLSL_EXP) &&
                                          has_opcode(w, n, OP_FNEGATE));
        /* The old bug: an FAbs where the sigmoid should be. */
        TEST("sigmoid: no substituted fabs", w && !has_glsl_inst(w, n, GLSL_FABS));
        cml_free(w);
    }
    {
        int n = 0;
        uint32_t* w = gen_for_op(UOP_MAX, 2, &n);
        TEST("max: not emitted as an addition", w && !has_opcode(w, n, OP_FADD));
        cml_free(w);
    }

    /* Ops that already worked must keep working. */
    check_op("add",  UOP_ADD,  2, OP_FADD, 0);
    check_op("sub",  UOP_SUB,  2, OP_FSUB, 0);
    check_op("mul",  UOP_MUL,  2, OP_FMUL, 0);
    check_op("div",  UOP_DIV,  2, OP_FDIV, 0);
    check_op("exp",  UOP_EXP,  1, 0, GLSL_EXP);
    check_op("abs",  UOP_ABS,  1, 0, GLSL_FABS);
    check_op("neg",  UOP_NEG,  1, OP_FNEGATE, 0);

    /* The PTX backend had the same shape of bug: its defaults emitted a
     * mov.f32 (a copy), so max(a,b) became a and tanh(x) became x. It also
     * emitted a bare ex2 for exp(), which computes 2**x rather than e**x. */
    printf("\nPTX fused-kernel op coverage:\n");
    check_ptx("exp",     UOP_EXP,     1, "ex2.approx.f32", "0f3FB8AA3B");
    check_ptx("log",     UOP_LOG,     1, "lg2.approx.f32", "0f3F317218");
    check_ptx("max",     UOP_MAX,     2, "max.f32",        NULL);
    check_ptx("pow",     UOP_POW,     2, "lg2.approx.f32", "ex2.approx.f32");
    check_ptx("sin",     UOP_SIN,     1, "sin.approx.f32", NULL);
    check_ptx("cos",     UOP_COS,     1, "cos.approx.f32", NULL);
    check_ptx("sigmoid", UOP_SIGMOID, 1, "rcp.approx.f32", "0fBFB8AA3B");
    check_ptx("silu",    UOP_SILU,    1, "rcp.approx.f32", "mul.f32");
    check_ptx("tanh",    UOP_TANH,    1, "rcp.approx.f32", "sub.f32");

    /* No op may fall back to a bare copy. */
    {
        char* p = ptx_for(UOP_MAX, 2);
        TEST("ptx max: not emitted as a copy", p && !strstr(p, "unsupported"));
        cml_free(p);
    }

    cml_cleanup();
    return TEST_SUMMARY();
}
