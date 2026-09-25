/* Numeric validation of the PTX codegen, without a GPU.
 *
 * `test_ptx_codegen.c` checks that the emitted PTX *text* mentions the right
 * instructions. That catches a missing op but not a wrong one: a kernel that
 * reads the wrong register, swaps two operands, or picks the wrong rounding mode
 * emits perfectly plausible text. Confirming the numbers needed an NVIDIA GPU,
 * and CI has none -- so `docs/REMAINING_WORK.md` carried "codegen exists but
 * numeric validation needs hardware" as an open item.
 *
 * It does not, for this class of kernel. The emitted elementwise kernels are
 * straight-line scalar f32 with a bounds guard, so tests/ptx_interp.h executes
 * them on the host and this suite compares the result against the same reference
 * the CPU path computes. A GPU is still required to validate the *driver* path
 * (NVRTC, module load, launch); that is a different claim and stays open.
 */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ops/ir/gpu/ptx_codegen.h"
#include "ops/uops.h"
#include "alloc/cml_allocator.h"
#include "ptx_interp.h"
#include "test_harness.h"

#define N 37 /* deliberately not a multiple of the block size */
#define BLOCK 8
#define TOL 2e-5f

static CMLPTXCodegen* g_cg;

static bool close_enough(float got, float want) {
    if (isnan(want))
        return isnan(got);
    if (isinf(want))
        return isinf(got) && ((want > 0) == (got > 0));
    float denom = fmaxf(1.0f, fmaxf(fabsf(got), fabsf(want)));
    return fabsf(got - want) / denom <= TOL;
}

/* ---- unary ops ----------------------------------------------------------- */

typedef struct {
    UOpType op;
    const char* name;
    float (*ref)(float);
    float lo, hi;
} UnaryCase;

static float r_neg(float x) { return -x; }
static float r_abs(float x) { return fabsf(x); }
static float r_sqrt(float x) { return sqrtf(x); }
static float r_rsqrt(float x) { return 1.0f / sqrtf(x); }
static float r_recip(float x) { return 1.0f / x; }
static float r_exp(float x) { return expf(x); }
static float r_log(float x) { return logf(x); }
static float r_exp2(float x) { return exp2f(x); }
static float r_log2(float x) { return log2f(x); }
static float r_sin(float x) { return sinf(x); }
static float r_cos(float x) { return cosf(x); }
static float r_tan(float x) { return tanf(x); }
static float r_tanh(float x) { return tanhf(x); }
static float r_square(float x) { return x * x; }
static float r_floor(float x) { return floorf(x); }
static float r_ceil(float x) { return ceilf(x); }
static float r_sign(float x) { return x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f); }
static float r_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float r_relu6(float x) { return fminf(fmaxf(x, 0.0f), 6.0f); }
static float r_hard_sigmoid(float x) { return fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f); }
static float r_hard_tanh(float x) { return fminf(fmaxf(x, -1.0f), 1.0f); }
static float r_silu(float x) { return x * r_sigmoid(x); }
static float r_quick_gelu(float x) { return x * r_sigmoid(1.702f * x); }
static float r_leaky_relu(float x) { return x >= 0 ? x : 0.01f * x; }
static float r_elu(float x) { return x >= 0 ? x : (expf(x) - 1.0f); }
static float r_mish(float x) { return x * tanhf(logf(1.0f + expf(x))); }
static float r_hardswish(float x) { return x * r_hard_sigmoid(x); }
/* tanh-approximation GELU, matching execution_typed.c */
static float r_gelu(float x) {
    const float k = 0.7978845608028654f; /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

static const UnaryCase UNARY[] = {
    {UOP_NEG, "neg", r_neg, -3.0f, 3.0f},
    {UOP_ABS, "abs", r_abs, -3.0f, 3.0f},
    {UOP_SQRT, "sqrt", r_sqrt, 0.25f, 9.0f},
    {UOP_RSQRT, "rsqrt", r_rsqrt, 0.25f, 9.0f},
    {UOP_RECIP, "recip", r_recip, 0.5f, 4.0f},
    {UOP_EXP, "exp", r_exp, -2.0f, 2.0f},
    {UOP_LOG, "log", r_log, 0.25f, 5.0f},
    {UOP_EXP2, "exp2", r_exp2, -2.0f, 2.0f},
    {UOP_LOG2, "log2", r_log2, 0.25f, 5.0f},
    {UOP_SIN, "sin", r_sin, -3.0f, 3.0f},
    {UOP_COS, "cos", r_cos, -3.0f, 3.0f},
    {UOP_TAN, "tan", r_tan, -1.0f, 1.0f},
    {UOP_TANH, "tanh", r_tanh, -2.0f, 2.0f},
    {UOP_SQUARE, "square", r_square, -3.0f, 3.0f},
    {UOP_FLOOR, "floor", r_floor, -3.0f, 3.0f},
    {UOP_CEIL, "ceil", r_ceil, -3.0f, 3.0f},
    {UOP_SIGN, "sign", r_sign, -3.0f, 3.0f},
    {UOP_SIGMOID, "sigmoid", r_sigmoid, -3.0f, 3.0f},
    {UOP_RELU6, "relu6", r_relu6, -2.0f, 8.0f},
    {UOP_HARD_SIGMOID, "hard_sigmoid", r_hard_sigmoid, -4.0f, 4.0f},
    {UOP_HARD_TANH, "hard_tanh", r_hard_tanh, -3.0f, 3.0f},
    {UOP_SILU, "silu", r_silu, -3.0f, 3.0f},
    {UOP_QUICK_GELU, "quick_gelu", r_quick_gelu, -3.0f, 3.0f},
    {UOP_LEAKY_RELU, "leaky_relu", r_leaky_relu, -3.0f, 3.0f},
    {UOP_ELU, "elu", r_elu, -2.0f, 2.0f},
    {UOP_MISH, "mish", r_mish, -2.0f, 2.0f},
    {UOP_HARDSWISH, "hardswish", r_hardswish, -4.0f, 4.0f},
    {UOP_GELU, "gelu", r_gelu, -2.0f, 2.0f},
};

/* ---- binary ops ---------------------------------------------------------- */

typedef struct {
    UOpType op;
    const char* name;
    float (*ref)(float, float);
} BinaryCase;

static float r_add(float a, float b) { return a + b; }
static float r_sub(float a, float b) { return a - b; }
static float r_mul(float a, float b) { return a * b; }
static float r_div(float a, float b) { return a / b; }
static float r_max(float a, float b) { return fmaxf(a, b); }
static float r_min(float a, float b) { return fminf(a, b); }
static float r_cmplt(float a, float b) { return a < b ? 1.0f : 0.0f; }
static float r_cmpgt(float a, float b) { return a > b ? 1.0f : 0.0f; }
static float r_cmple(float a, float b) { return a <= b ? 1.0f : 0.0f; }
static float r_cmpge(float a, float b) { return a >= b ? 1.0f : 0.0f; }
static float r_cmpeq(float a, float b) { return a == b ? 1.0f : 0.0f; }
static float r_cmpne(float a, float b) { return a != b ? 1.0f : 0.0f; }

static const BinaryCase BINARY[] = {
    {UOP_ADD, "add", r_add},       {UOP_SUB, "sub", r_sub},       {UOP_MUL, "mul", r_mul},
    {UOP_DIV, "div", r_div},       {UOP_MAX, "max", r_max},       {UOP_MINIMUM, "minimum", r_min},
    {UOP_CMPLT, "cmplt", r_cmplt}, {UOP_CMPGT, "cmpgt", r_cmpgt}, {UOP_CMPLE, "cmple", r_cmple},
    {UOP_CMPGE, "cmpge", r_cmpge}, {UOP_CMPEQ, "cmpeq", r_cmpeq}, {UOP_CMPNE, "cmpne", r_cmpne},
};

/* ---- drivers ------------------------------------------------------------- */

static int run_unary(const UnaryCase* c) {
    char* ptx = cml_ptx_gen_unary(g_cg, c->op, "k");
    if (!ptx) {
        printf("(codegen returned NULL) ");
        return 0;
    }

    float in[N], out[N];
    for (int i = 0; i < N; i++) {
        in[i]  = c->lo + (c->hi - c->lo) * ((float)i + 0.5f) / (float)N;
        out[i] = -12345.0f; /* poison: an unwritten element must be caught */
    }

    PtxiState st;
    memset(&st, 0, sizeof(st));
    ptxi_add_buffer(&st, "in", in, N);
    ptxi_add_buffer(&st, "out", out, N);

    int ok = 1;
    if (!ptxi_run(&st, ptx, N, BLOCK)) {
        printf("(%s) ", st.err);
        ok = 0;
    }
    for (int i = 0; i < N && ok; i++) {
        float want = c->ref(in[i]);
        if (!close_enough(out[i], want)) {
            printf("(i=%d in=%g got=%g want=%g) ", i, (double)in[i], (double)out[i], (double)want);
            ok = 0;
        }
    }
    cml_free(ptx);
    return ok;
}

static int run_binary(const BinaryCase* c) {
    char* ptx = cml_ptx_gen_binary(g_cg, c->op, "k");
    if (!ptx) {
        printf("(codegen returned NULL) ");
        return 0;
    }

    float a[N], b[N], out[N];
    for (int i = 0; i < N; i++) {
        a[i] = -2.0f + 4.0f * ((float)i + 0.5f) / (float)N;
        /* every third element equal, so the equality comparisons see both sides */
        b[i]   = (i % 3 == 0) ? a[i] : (2.5f - 3.0f * ((float)i + 0.25f) / (float)N);
        out[i] = -12345.0f;
    }

    PtxiState st;
    memset(&st, 0, sizeof(st));
    ptxi_add_buffer(&st, "a", a, N);
    ptxi_add_buffer(&st, "b", b, N);
    ptxi_add_buffer(&st, "out", out, N);

    int ok = 1;
    if (!ptxi_run(&st, ptx, N, BLOCK)) {
        printf("(%s) ", st.err);
        ok = 0;
    }
    for (int i = 0; i < N && ok; i++) {
        float want = c->ref(a[i], b[i]);
        if (!close_enough(out[i], want)) {
            printf("(i=%d a=%g b=%g got=%g want=%g) ", i, (double)a[i], (double)b[i],
                   (double)out[i], (double)want);
            ok = 0;
        }
    }
    cml_free(ptx);
    return ok;
}

static int test_fill_value(void) {
    char* ptx = cml_ptx_gen_fill(g_cg, 3.25f, "k");
    if (!ptx) {
        printf("(codegen returned NULL) ");
        return 0;
    }
    float out[N];
    for (int i = 0; i < N; i++)
        out[i] = -12345.0f;

    PtxiState st;
    memset(&st, 0, sizeof(st));
    ptxi_add_buffer(&st, "out", out, N);

    int ok = ptxi_run(&st, ptx, N, BLOCK);
    if (!ok)
        printf("(%s) ", st.err);
    for (int i = 0; i < N && ok; i++)
        if (!close_enough(out[i], 3.25f)) {
            printf("(i=%d got=%g want=3.25) ", i, (double)out[i]);
            ok = 0;
        }
    cml_free(ptx);
    return ok;
}

/* The guard must leave elements past n untouched: grid*block > n here (40 > 37),
 * so a missing or inverted `setp.ge.u32` bounds check would scribble past the
 * end -- which on a real GPU is a memory fault, and is exactly the kind of bug
 * text inspection cannot see. */
static int test_bounds_guard(void) {
    char* ptx = cml_ptx_gen_unary(g_cg, UOP_NEG, "k");
    if (!ptx)
        return 0;

    const int over = N + BLOCK; /* room for the tail threads to misbehave into */
    float in[N + BLOCK], out[N + BLOCK];
    for (int i = 0; i < over; i++) {
        in[i]  = 1.0f;
        out[i] = 999.0f;
    }

    PtxiState st;
    memset(&st, 0, sizeof(st));
    /* Buffers are declared oversized so an out-of-range store is observable as a
     * changed value rather than an interpreter error. */
    ptxi_add_buffer(&st, "in", in, (size_t)over);
    ptxi_add_buffer(&st, "out", out, (size_t)over);

    int ok = ptxi_run(&st, ptx, N, BLOCK);
    if (!ok)
        printf("(%s) ", st.err);
    for (int i = N; i < over && ok; i++)
        if (out[i] != 999.0f) {
            printf("(guard leaked: out[%d]=%g) ", i, (double)out[i]);
            ok = 0;
        }
    cml_free(ptx);
    return ok;
}

/* The interpreter must refuse what it does not implement, so a future codegen
 * change cannot quietly pass by emitting instructions nobody executes. */
static int test_interpreter_rejects_unknown(void) {
    static const char* bogus = ".visible .entry k(\n"
                               "    .param .u64 param_out,\n"
                               "    .param .u32 param_n\n"
                               ") {\n"
                               "    mov.u32 %r0, %tid.x;\n"
                               "    wmma.load.a.sync.aligned.m16n16k16 %f0, [%rd0];\n"
                               "    ret;\n"
                               "}\n";
    float out[4]             = {0, 0, 0, 0};
    PtxiState st;
    memset(&st, 0, sizeof(st));
    ptxi_add_buffer(&st, "out", out, 4);
    return ptxi_run(&st, bogus, 4, 4) == false && strstr(st.err, "unsupported") != NULL;
}

int main(void) {
    printf("PTX Numeric Validation (interpreted; no GPU required)\n\n");

    g_cg = cml_ptx_codegen_create(75, NULL);
    if (!g_cg) {
        printf("could not create PTX codegen\n");
        return 1;
    }

    printf("Interpreter self-checks:\n");
    TEST(interpreter_rejects_unknown);
    TEST(bounds_guard);
    TEST(fill_value);

    printf("\nUnary ops vs CPU reference:\n");
    for (size_t i = 0; i < sizeof(UNARY) / sizeof(UNARY[0]); i++) {
        tests_run++;
        printf("  %-55s ", UNARY[i].name);
        fflush(stdout);
        if (run_unary(&UNARY[i])) {
            tests_passed++;
            printf("[PASS]\n");
        } else {
            printf("[FAIL]\n");
        }
    }

    printf("\nBinary ops vs CPU reference:\n");
    for (size_t i = 0; i < sizeof(BINARY) / sizeof(BINARY[0]); i++) {
        tests_run++;
        printf("  %-55s ", BINARY[i].name);
        fflush(stdout);
        if (run_binary(&BINARY[i])) {
            tests_passed++;
            printf("[PASS]\n");
        } else {
            printf("[FAIL]\n");
        }
    }

    cml_ptx_codegen_destroy(g_cg);
    return TEST_SUMMARY();
}
