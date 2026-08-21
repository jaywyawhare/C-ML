/* The NVIDIA userspace driver's PTX generator, checked without a GPU.
 *
 * It was reachable only through cml_nv_execute_graph, which needs real
 * hardware, so nothing ever looked at what it emitted. Three bugs had
 * accumulated:
 *
 *   1. Register names were written "%%f1" in strings that are inserted through
 *      a "%s" argument, so they are NOT format-processed and reached the
 *      assembler literally as "%%f1". Every kernel it produced was malformed.
 *      (The surrounding template IS the format string, so it correctly uses %%.)
 *   2. The case labels admitted TANH, SIGMOID and SILU but the inner switch had
 *      no arm for them, so all three fell to `default: mov.f32` -- a copy. On
 *      this path tanh(x), sigmoid(x) and silu(x) each compiled to x.
 *   3. exp emitted a bare ex2 (2**x, not e**x) and log a bare lg2 (log2, not
 *      ln), each missing its scale factor.
 *
 * Generating the text and inspecting it costs nothing and catches all three.
 */

#include "cml.h"
#include "test_require.h"
#include "ops/ir/ir.h"
#include "ops/ir/gpu/nv_driver.h"
#include <stdio.h>
#include <string.h>

static int checks = 0, failures = 0;

static void fail(const char* name, const char* why) {
    printf("  %-12s %s\n", name, why);
    failures++;
}

/* Build a one-node graph for `op` and return its PTX. */
static char* ptx_for(UOpType op, int nargs) {
    TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                      .has_dtype = true, .has_device = true};
    int shape[1] = {8};
    Tensor* a = tensor_zeros(shape, 1, &c);
    Tensor* r = NULL;
    if (nargs == 1) {
        switch (op) {
        case UOP_NEG:     r = uop_neg(a); break;
        case UOP_EXP:     r = uop_exp(a); break;
        case UOP_LOG:     r = uop_log(a); break;
        case UOP_SQRT:    r = uop_sqrt(a); break;
        case UOP_ABS:     r = uop_abs(a); break;
        case UOP_SIN:     r = uop_sin(a); break;
        case UOP_COS:     r = uop_cos(a); break;
        case UOP_RECIP:   r = uop_recip(a); break;
        case UOP_TANH:    r = uop_tanh(a); break;
        case UOP_SIGMOID: r = uop_sigmoid(a); break;
        case UOP_SILU:    r = uop_silu(a); break;
        default: break;
        }
    } else {
        Tensor* b = tensor_zeros(shape, 1, &c);
        switch (op) {
        case UOP_ADD: r = uop_add(a, b); break;
        case UOP_SUB: r = uop_sub(a, b); break;
        case UOP_MUL: r = uop_mul(a, b); break;
        case UOP_DIV: r = uop_div(a, b); break;
        default: break;
        }
    }
    if (!r || !r->ir_node) return NULL;
    return cml_nv_gen_ptx_for_node((struct IRNode*)r->ir_node, 50);
}

/* `need` must appear; `banned` (if given) must not. */
static void check(const char* name, UOpType op, int nargs,
                  const char* need, const char* banned) {
    checks++;
    char* ptx = ptx_for(op, nargs);
    if (!ptx) {
        fail(name, "generator returned NULL");
        cml_reset_ir_context();
        return;
    }
    /* No kernel may contain a doubled percent: that is invalid PTX. */
    if (strstr(ptx, "%%") != NULL)
        fail(name, "emits '%%' -- register names are not valid PTX");
    else if (need && !strstr(ptx, need))
        fail(name, "missing expected instruction/constant");
    else if (banned && strstr(ptx, banned))
        fail(name, "emits the substituted instruction instead of the real one");
    cml_free(ptx);
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("NVIDIA driver PTX generation:\n");

    /* Ops that always had an arm -- they must still be right, and unescaped. */
    check("neg",     UOP_NEG,     1, "neg.f32",          NULL);
    check("sqrt",    UOP_SQRT,    1, "sqrt.approx.f32",  NULL);
    check("abs",     UOP_ABS,     1, "abs.f32",          NULL);
    check("sin",     UOP_SIN,     1, "sin.approx.f32",   NULL);
    check("cos",     UOP_COS,     1, "cos.approx.f32",   NULL);
    /* uop_recip composes as 1/x, so it reaches the generator as a DIV node --
     * the UOP_RECIP arm exists but is not reachable from the public op. */
    check("recip(->div)", UOP_RECIP, 1, "div.approx.f32", NULL);

    /* exp/log need their scale factors: ex2 alone is 2**x, lg2 alone is log2. */
    check("exp",     UOP_EXP,     1, "0f3FB8AA3B",       NULL);  /* log2(e) */
    check("log",     UOP_LOG,     1, "0f3F317218",       NULL);  /* ln(2)   */

    /* These three used to compile to a bare copy. */
    check("tanh",    UOP_TANH,    1, "rcp.approx.f32",   "mov.f32");
    check("sigmoid", UOP_SIGMOID, 1, "rcp.approx.f32",   "mov.f32");
    check("silu",    UOP_SILU,    1, "mul.f32",          "mov.f32");

    check("add",     UOP_ADD,     2, "add.f32",          NULL);
    check("sub",     UOP_SUB,     2, "sub.f32",          NULL);
    check("mul",     UOP_MUL,     2, "mul.f32",          NULL);
    check("div",     UOP_DIV,     2, "div.approx.f32",   NULL);

    printf("\n%d checks, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
