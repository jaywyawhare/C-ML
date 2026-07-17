/* Decompose pass: composite ops (relu, sigmoid, gelu, softmax, var, ...) must be
 * lowered to the minimal primitive UOP set before execution, and the decomposed
 * graph must compute the same values as the composite. */
#include <stdio.h>
#include <math.h>
#include "cml.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/decompose.h"
#include "ops/ir/context.h"

static int g_fail = 0;
#define CHECK(name, cond) do { \
    if (cond) printf("  PASS  %s\n", name); \
    else { printf("  FAIL  %s\n", name); g_fail = 1; } } while (0)

/* Composite UOPs that MUST NOT survive decomposition. */
static int is_composite(UOpType t) {
    switch (t) {
    case UOP_RELU: case UOP_SIGMOID: case UOP_GELU:
    case UOP_TANH: case UOP_SILU: case UOP_ELU: case UOP_SELU:
    case UOP_VAR: case UOP_STD: case UOP_MAXPOOL2D: case UOP_AVGPOOL2D:
    case UOP_CONV2D: case UOP_SUB: case UOP_DIV: case UOP_NEG: case UOP_FLATTEN:
        return 1;
    default: return 0;
    }
}

static int count_composites(CMLGraph_t ir) {
    int n = 0;
    for (struct IRNode* p = ir->head; p; p = p->next)
        if (is_composite(p->type)) n++;
    return n;
}

int main(void) {
    cml_init();
    printf("=== Decompose pass ===\n");

    /* build a graph of composites, decompose, and assert none survive */
    cml_reset_ir_context();
    CMLGraph_t ir = cml_ir_get_or_create_context();
    float xd[8] = {-1.0f, -0.3f, 0.2f, 0.7f, 1.5f, -0.8f, 0.1f, 0.9f};
    Tensor* x = cml_tensor_1d(xd, 8);
    Tensor* a = cml_relu(x);
    Tensor* b = cml_sigmoid(a);
    Tensor* c = cml_silu(b);
    (void)c;
    int before = count_composites(ir);
    cml_ir_decompose(ir);
    int after = count_composites(ir);
    printf("    composite nodes: before=%d after=%d\n", before, after);
    CHECK("graph had composite ops before decompose", before > 0);
    CHECK("no composite ops survive decompose (lowered to primitives)", after == 0);

    /* value correctness through the decomposed+executed path: relu then sigmoid */
    cml_reset_ir_context();
    float yd[5] = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    Tensor* y  = cml_tensor_1d(yd, 5);
    Tensor* ry = cml_relu(y);
    float* r = (float*)tensor_data_ptr(ry);
    float relu_ref[5] = {0, 0, 0, 0.5f, 2.0f};
    int ok = 1; for (int i = 0; i < 5; i++) if (fabsf(r[i] - relu_ref[i]) > 1e-5f) ok = 0;
    CHECK("relu value correct through decomposed path", ok);

    cml_reset_ir_context();
    Tensor* y2 = cml_tensor_1d(yd, 5);
    Tensor* sy = cml_sigmoid(y2);
    float* s = (float*)tensor_data_ptr(sy);
    int ok2 = 1; for (int i = 0; i < 5; i++) {
        float ref = 1.0f / (1.0f + expf(-yd[i]));
        if (fabsf(s[i] - ref) > 1e-4f) ok2 = 0;
    }
    CHECK("sigmoid value correct through decomposed path", ok2);

    cml_reset_ir_context();
    printf("\n%s\n", g_fail ? "DECOMPOSE TESTS FAILED" : "All decompose tests passed");
    return g_fail;
}
