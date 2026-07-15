#include "ops/ir/aot.h"
#include "ops/ir/internal.h"
#include "ops/ir/context.h"
#include "core/logging.h"
#include "nn.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#include <ctype.h>
#include <unistd.h>

#ifdef CML_HAS_LLVM_BACKEND
#include "ops/ir/llvm/llvm_backend.h"
#endif

/* Rejects strings containing shell metacharacters that could allow
 * arbitrary command execution when passed to popen/system. */
static bool aot_validate_path(const char* path) __attribute__((unused));
static bool aot_validate_path(const char* path) {
    if (!path) return false;
    for (const char* p = path; *p; p++) {
        /* Reject shell metacharacters */
        if (*p == ';' || *p == '|' || *p == '&' || *p == '$' ||
            *p == '`' || *p == '\n' || *p == '\r' ||
            *p == '(' || *p == ')' || *p == '{' || *p == '}' ||
            *p == '<' || *p == '>' || *p == '!' || *p == '\\' ||
            *p == '\'' || *p == '"') {
            LOG_ERROR("Unsafe character '%c' in path: %s", *p, path);
            return false;
        }
    }
    return true;
}

AOTCompileOptions cml_aot_default_options(void) {
    AOTCompileOptions opts = {
        .target_triple = NULL,
        .cpu = NULL,
        .features = NULL,
        .opt_level = AOT_OPT_O3,
        .include_weights = false,
        .generate_header = true,
        .function_name = "cml_model_forward",
        .format = AOT_FORMAT_SHARED_LIB,
        .position_independent = true
    };
    return opts;
}

#ifdef CML_HAS_LLVM_BACKEND

/* ---- AOT C-codegen: value-numbered buffer map --------------------------
 * Each IR node produces one output tensor. A tensor consumed but never
 * produced is a graph input; the tail node's output is the graph output;
 * every other produced tensor is an intermediate that gets its own heap
 * buffer inside the generated function. The generated ABI is
 *     void FUNC(MemRef** inputs, MemRef** outputs)
 * so it works for any number of inputs (the old code only handled 2 or 3
 * and had every node clobber a single shared output buffer). */

enum { AOT_KIND_INPUT = 0, AOT_KIND_OUTPUT = 1, AOT_KIND_INTERMEDIATE = 2 };

typedef struct {
    Tensor* t;
    int     kind;
    int     index; /* input index, or intermediate index */
} AotBuf;

typedef struct {
    AotBuf* items;
    int     count;
    int     cap;
} AotBufMap;

static AotBuf* aot_map_find(AotBufMap* m, Tensor* t) {
    for (int i = 0; i < m->count; i++)
        if (m->items[i].t == t) return &m->items[i];
    return NULL;
}

static AotBuf* aot_map_add(AotBufMap* m, Tensor* t, int kind, int index) {
    if (m->count == m->cap) {
        int nc = m->cap ? m->cap * 2 : 16;
        AotBuf* ni = realloc(m->items, (size_t)nc * sizeof(AotBuf));
        if (!ni) return NULL;
        m->items = ni;
        m->cap = nc;
    }
    m->items[m->count].t = t;
    m->items[m->count].kind = kind;
    m->items[m->count].index = index;
    return &m->items[m->count++];
}

/* Writes the C pointer expression for tensor t into buf. */
static void aot_ptr_expr(AotBufMap* m, Tensor* t, char* buf, size_t n) {
    AotBuf* b = t ? aot_map_find(m, t) : NULL;
    if (!b) {
        snprintf(buf, n, "((float*)0)");
        return;
    }
    switch (b->kind) {
    case AOT_KIND_INPUT:  snprintf(buf, n, "inputs[%d]->aligned", b->index); break;
    case AOT_KIND_OUTPUT: snprintf(buf, n, "outputs[0]->aligned"); break;
    default:              snprintf(buf, n, "t%d", b->index); break;
    }
}

/* Emit `for (i) { float x = A[i]; O[i] = <expr>; }` */
static void aot_emit_unary(FILE* cf, const char* name, const char* O,
                           const char* A, int64_t n, const char* expr) {
    fprintf(cf, "    /* %s */\n", name);
    fprintf(cf, "    for (int64_t i = 0; i < %lld; i++) "
                "{ float x = %s[i]; %s[i] = %s; }\n",
            (long long)n, A, O, expr);
}

/* Emit an elementwise binary op with numpy-style scalar broadcasting: a
 * numel-1 operand is read as [0], otherwise as [i]. */
static void aot_emit_binary(FILE* cf, const char* name, const char* O, const char* A,
                            const char* B, int64_t na, int64_t nb, int64_t n,
                            const char* expr) {
    const char* ai = (na == 1) ? "0" : "i";
    const char* bi = (nb == 1) ? "0" : "i";
    fprintf(cf, "    /* %s */\n", name);
    fprintf(cf, "    for (int64_t i = 0; i < %lld; i++) "
                "{ float a = %s[%s]; float b = %s[%s]; %s[i] = %s; }\n",
            (long long)n, A, ai, B, bi, O, expr);
}

/* Emit a reduction over `outer x dim x inner` (contiguous input). */
static void aot_emit_reduce(FILE* cf, const char* name, const char* O, const char* A,
                            int64_t outer, int64_t dim, int64_t inner,
                            const char* init, const char* acc, const char* final_) {
    fprintf(cf, "    /* %s */\n", name);
    fprintf(cf, "    for (int64_t o = 0; o < %lld; o++)\n", (long long)outer);
    fprintf(cf, "      for (int64_t k = 0; k < %lld; k++) {\n", (long long)inner);
    fprintf(cf, "        float acc = %s;\n", init);
    fprintf(cf, "        for (int64_t j = 0; j < %lld; j++) "
                "{ float v = %s[(o * %lld + j) * %lld + k]; acc = %s; }\n",
            (long long)dim, A, (long long)dim, (long long)inner, acc);
    fprintf(cf, "        %s[o * %lld + k] = %s;\n", O, (long long)inner, final_);
    fprintf(cf, "      }\n");
}

#endif /* CML_HAS_LLVM_BACKEND */

int cml_aot_compile(CMLGraph_t ir, const char* output_path, const AOTCompileOptions* options) {
#ifdef CML_HAS_LLVM_BACKEND
    if (!ir || !output_path) {
        LOG_ERROR("Invalid arguments to cml_aot_compile");
        return -1;
    }

    /* Validate all user-supplied paths against command injection */
    if (!aot_validate_path(output_path)) {
        LOG_ERROR("Unsafe output path rejected");
        return -1;
    }

    AOTCompileOptions opts = options ? *options : cml_aot_default_options();

    if (opts.target_triple && !aot_validate_path(opts.target_triple)) {
        LOG_ERROR("Unsafe target_triple rejected");
        return -1;
    }
    if (opts.cpu && !aot_validate_path(opts.cpu)) {
        LOG_ERROR("Unsafe cpu option rejected");
        return -1;
    }

    const char* func_name = opts.function_name ? opts.function_name : "cml_model_forward";

    LOG_INFO("AOT compiling IR graph to: %s (C backend)", output_path);

    if (opts.format == AOT_FORMAT_HEADER_ONLY) {
        return cml_aot_generate_header(ir, output_path, func_name);
    }

    /* Build C source in a temporary file */
    char tmp_c_path[512];
    snprintf(tmp_c_path, sizeof(tmp_c_path), "/tmp/cml_aot_%d.c", (int)getpid());

    FILE* cf = fopen(tmp_c_path, "w");
    if (!cf) {
        LOG_ERROR("Failed to create temporary C file: %s", tmp_c_path);
        return -1;
    }

    /* --- Build the buffer map ------------------------------------------- */
    AotBufMap map = {0};

    struct IRNode* tail = NULL;
    for (struct IRNode* nd = ir->head; nd; nd = nd->next) tail = nd;
    if (!tail || !tail->output) {
        LOG_ERROR("AOT: empty graph or tail node has no output tensor");
        fclose(cf);
        remove(tmp_c_path);
        free(map.items);
        return -1;
    }

    /* Every node output is (provisionally) an intermediate. */
    for (struct IRNode* nd = ir->head; nd; nd = nd->next) {
        if (nd->output && !aot_map_find(&map, nd->output))
            aot_map_add(&map, nd->output, AOT_KIND_INTERMEDIATE, -1);
    }

    /* Graph inputs: tensors consumed but never produced, in first-use order. */
    int num_inputs = 0;
    for (struct IRNode* nd = ir->head; nd; nd = nd->next) {
        for (int k = 0; k < nd->num_inputs; k++) {
            Tensor* in = (nd->inputs) ? nd->inputs[k] : NULL;
            if (in && !aot_map_find(&map, in))
                aot_map_add(&map, in, AOT_KIND_INPUT, num_inputs++);
        }
    }

    /* The tail node's output is the single graph output. */
    {
        AotBuf* ob = aot_map_find(&map, tail->output);
        if (ob) { ob->kind = AOT_KIND_OUTPUT; ob->index = 0; }
    }

    /* Number the remaining intermediates. */
    int num_intermediates = 0;
    for (int i = 0; i < map.count; i++)
        if (map.items[i].kind == AOT_KIND_INTERMEDIATE)
            map.items[i].index = num_intermediates++;

    /* --- Emit preamble -------------------------------------------------- */
    fprintf(cf, "/* Auto-generated by CML AOT compiler */\n");
    fprintf(cf, "#include <stdint.h>\n");
    fprintf(cf, "#include <math.h>\n");
    fprintf(cf, "#include <string.h>\n");
    fprintf(cf, "#include <stdlib.h>\n\n");

    fprintf(cf, "typedef struct {\n");
    fprintf(cf, "    float* allocated;\n");
    fprintf(cf, "    float* aligned;\n");
    fprintf(cf, "    int64_t offset;\n");
    fprintf(cf, "    int64_t sizes[8];\n");
    fprintf(cf, "    int64_t strides[8];\n");
    fprintf(cf, "} MemRef;\n\n");

    fprintf(cf, "static inline float _cml_sigmoid(float x){ return 1.0f/(1.0f+expf(-x)); }\n");
    fprintf(cf, "static inline float _cml_softplus(float x){ return log1pf(expf(x)); }\n\n");

    /* --- Function signature --------------------------------------------- */
    fprintf(cf, "void %s(MemRef** inputs, MemRef** outputs) {\n", func_name);
    fprintf(cf, "    (void)inputs; (void)outputs;\n");

    /* Allocate intermediate buffers. */
    for (int i = 0; i < map.count; i++) {
        if (map.items[i].kind != AOT_KIND_INTERMEDIATE) continue;
        int64_t nm = map.items[i].t ? (int64_t)map.items[i].t->numel : 0;
        if (nm < 1) nm = 1;
        fprintf(cf, "    float* t%d = (float*)malloc(%lld * sizeof(float));\n",
                map.items[i].index, (long long)nm);
    }
    if (num_intermediates > 0) {
        fprintf(cf, "    if (");
        int first = 1;
        for (int i = 0; i < map.count; i++) {
            if (map.items[i].kind != AOT_KIND_INTERMEDIATE) continue;
            fprintf(cf, "%s!t%d", first ? "" : " || ", map.items[i].index);
            first = 0;
        }
        fprintf(cf, ") goto _cml_cleanup;\n");
    }

    /* --- Emit one statement per node ------------------------------------ */
    bool emit_ok = true;
    for (struct IRNode* node = ir->head; node && emit_ok; node = node->next) {
        if (!node->output) continue;

        char o[64], a[64], b[64];
        aot_ptr_expr(&map, node->output, o, sizeof(o));
        int64_t n = (int64_t)node->output->numel;

        Tensor* in0 = (node->num_inputs > 0 && node->inputs) ? node->inputs[0] : NULL;
        Tensor* in1 = (node->num_inputs > 1 && node->inputs) ? node->inputs[1] : NULL;
        int64_t na = in0 ? (int64_t)in0->numel : 0;
        int64_t nb = in1 ? (int64_t)in1->numel : 0;
        if (in0) aot_ptr_expr(&map, in0, a, sizeof(a));
        if (in1) aot_ptr_expr(&map, in1, b, sizeof(b));

        /* Binary elementwise ops require two inputs. */
        bool needs_binary = false;
        switch (node->type) {
        case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
        case UOP_MAX: case UOP_MINIMUM: case UOP_POW: case UOP_MOD:
        case UOP_IDIV: case UOP_COPYSIGN: case UOP_LOGADDEXP:
        case UOP_CMPLT: case UOP_CMPLE: case UOP_CMPGT: case UOP_CMPGE:
        case UOP_CMPEQ: case UOP_CMPNE: case UOP_LOGICAL_AND: case UOP_LOGICAL_OR:
            needs_binary = true;
            break;
        default:
            break;
        }
        if (needs_binary && !in1) {
            LOG_ERROR("AOT: binary UOp %d is missing its second input", (int)node->type);
            emit_ok = false;
            break;
        }

        switch (node->type) {
        /* ---- Binary elementwise ---- */
        case UOP_ADD:      aot_emit_binary(cf, "ADD",      o, a, b, na, nb, n, "a + b"); break;
        case UOP_SUB:      aot_emit_binary(cf, "SUB",      o, a, b, na, nb, n, "a - b"); break;
        case UOP_MUL:      aot_emit_binary(cf, "MUL",      o, a, b, na, nb, n, "a * b"); break;
        case UOP_DIV:      aot_emit_binary(cf, "DIV",      o, a, b, na, nb, n, "a / b"); break;
        case UOP_MAX:      aot_emit_binary(cf, "MAX",      o, a, b, na, nb, n, "a > b ? a : b"); break;
        case UOP_MINIMUM:  aot_emit_binary(cf, "MINIMUM",  o, a, b, na, nb, n, "a < b ? a : b"); break;
        case UOP_POW:      aot_emit_binary(cf, "POW",      o, a, b, na, nb, n, "powf(a, b)"); break;
        case UOP_MOD:      aot_emit_binary(cf, "MOD",      o, a, b, na, nb, n, "fmodf(a, b)"); break;
        case UOP_IDIV:     aot_emit_binary(cf, "IDIV",     o, a, b, na, nb, n, "floorf(a / b)"); break;
        case UOP_COPYSIGN: aot_emit_binary(cf, "COPYSIGN", o, a, b, na, nb, n, "copysignf(a, b)"); break;
        case UOP_LOGADDEXP:aot_emit_binary(cf, "LOGADDEXP",o, a, b, na, nb, n,
                                           "fmaxf(a, b) + log1pf(expf(-fabsf(a - b)))"); break;
        case UOP_CMPLT:    aot_emit_binary(cf, "CMPLT",    o, a, b, na, nb, n, "a < b ? 1.0f : 0.0f"); break;
        case UOP_CMPLE:    aot_emit_binary(cf, "CMPLE",    o, a, b, na, nb, n, "a <= b ? 1.0f : 0.0f"); break;
        case UOP_CMPGT:    aot_emit_binary(cf, "CMPGT",    o, a, b, na, nb, n, "a > b ? 1.0f : 0.0f"); break;
        case UOP_CMPGE:    aot_emit_binary(cf, "CMPGE",    o, a, b, na, nb, n, "a >= b ? 1.0f : 0.0f"); break;
        case UOP_CMPEQ:    aot_emit_binary(cf, "CMPEQ",    o, a, b, na, nb, n, "a == b ? 1.0f : 0.0f"); break;
        case UOP_CMPNE:    aot_emit_binary(cf, "CMPNE",    o, a, b, na, nb, n, "a != b ? 1.0f : 0.0f"); break;
        case UOP_LOGICAL_AND: aot_emit_binary(cf, "LOGICAL_AND", o, a, b, na, nb, n,
                                              "(a != 0.0f && b != 0.0f) ? 1.0f : 0.0f"); break;
        case UOP_LOGICAL_OR:  aot_emit_binary(cf, "LOGICAL_OR",  o, a, b, na, nb, n,
                                              "(a != 0.0f || b != 0.0f) ? 1.0f : 0.0f"); break;

        /* ---- Unary elementwise ---- */
        case UOP_NEG:    aot_emit_unary(cf, "NEG",    o, a, n, "-x"); break;
        case UOP_EXP:    aot_emit_unary(cf, "EXP",    o, a, n, "expf(x)"); break;
        case UOP_LOG:    aot_emit_unary(cf, "LOG",    o, a, n, "logf(x)"); break;
        case UOP_SQRT:   aot_emit_unary(cf, "SQRT",   o, a, n, "sqrtf(x)"); break;
        case UOP_RECIP:  aot_emit_unary(cf, "RECIP",  o, a, n, "1.0f / x"); break;
        case UOP_ABS:    aot_emit_unary(cf, "ABS",    o, a, n, "fabsf(x)"); break;
        case UOP_SIN:    aot_emit_unary(cf, "SIN",    o, a, n, "sinf(x)"); break;
        case UOP_COS:    aot_emit_unary(cf, "COS",    o, a, n, "cosf(x)"); break;
        case UOP_TAN:    aot_emit_unary(cf, "TAN",    o, a, n, "tanf(x)"); break;
        case UOP_TANH:   aot_emit_unary(cf, "TANH",   o, a, n, "tanhf(x)"); break;
        case UOP_SIGMOID:aot_emit_unary(cf, "SIGMOID",o, a, n, "_cml_sigmoid(x)"); break;
        case UOP_SIGN:   aot_emit_unary(cf, "SIGN",   o, a, n, "(float)((x > 0.0f) - (x < 0.0f))"); break;
        case UOP_FLOOR:  aot_emit_unary(cf, "FLOOR",  o, a, n, "floorf(x)"); break;
        case UOP_CEIL:   aot_emit_unary(cf, "CEIL",   o, a, n, "ceilf(x)"); break;
        case UOP_ROUND:  aot_emit_unary(cf, "ROUND",  o, a, n, "roundf(x)"); break;
        case UOP_LOG2:   aot_emit_unary(cf, "LOG2",   o, a, n, "log2f(x)"); break;
        case UOP_EXP2:   aot_emit_unary(cf, "EXP2",   o, a, n, "exp2f(x)"); break;
        case UOP_ASIN:   aot_emit_unary(cf, "ASIN",   o, a, n, "asinf(x)"); break;
        case UOP_ACOS:   aot_emit_unary(cf, "ACOS",   o, a, n, "acosf(x)"); break;
        case UOP_ATAN:   aot_emit_unary(cf, "ATAN",   o, a, n, "atanf(x)"); break;
        case UOP_SQUARE: aot_emit_unary(cf, "SQUARE", o, a, n, "x * x"); break;
        case UOP_RSQRT:  aot_emit_unary(cf, "RSQRT",  o, a, n, "1.0f / sqrtf(x)"); break;
        case UOP_ERF:    aot_emit_unary(cf, "ERF",    o, a, n, "erff(x)"); break;
        case UOP_ERFC:   aot_emit_unary(cf, "ERFC",   o, a, n, "erfcf(x)"); break;
        case UOP_LOG10:  aot_emit_unary(cf, "LOG10",  o, a, n, "log10f(x)"); break;
        case UOP_SINH:   aot_emit_unary(cf, "SINH",   o, a, n, "sinhf(x)"); break;
        case UOP_COSH:   aot_emit_unary(cf, "COSH",   o, a, n, "coshf(x)"); break;
        case UOP_ASINH:  aot_emit_unary(cf, "ASINH",  o, a, n, "asinhf(x)"); break;
        case UOP_ACOSH:  aot_emit_unary(cf, "ACOSH",  o, a, n, "acoshf(x)"); break;
        case UOP_ATANH:  aot_emit_unary(cf, "ATANH",  o, a, n, "atanhf(x)"); break;
        case UOP_TRUNC:  aot_emit_unary(cf, "TRUNC",  o, a, n, "truncf(x)"); break;
        case UOP_ISINF:  aot_emit_unary(cf, "ISINF",  o, a, n, "isinf(x) ? 1.0f : 0.0f"); break;
        case UOP_ISNAN:  aot_emit_unary(cf, "ISNAN",  o, a, n, "isnan(x) ? 1.0f : 0.0f"); break;
        case UOP_ISFINITE:aot_emit_unary(cf,"ISFINITE",o,a, n, "isfinite(x) ? 1.0f : 0.0f"); break;
        case UOP_LOGICAL_NOT: aot_emit_unary(cf, "LOGICAL_NOT", o, a, n, "x == 0.0f ? 1.0f : 0.0f"); break;

        /* ---- Activations ---- */
        case UOP_RELU:       aot_emit_unary(cf, "RELU",       o, a, n, "x > 0.0f ? x : 0.0f"); break;
        case UOP_RELU6:      aot_emit_unary(cf, "RELU6",      o, a, n, "x < 0.0f ? 0.0f : (x > 6.0f ? 6.0f : x)"); break;
        case UOP_HARD_SIGMOID:aot_emit_unary(cf,"HARD_SIGMOID",o,a, n, "x < -3.0f ? 0.0f : (x > 3.0f ? 1.0f : (x + 3.0f) / 6.0f)"); break;
        case UOP_HARD_TANH:  aot_emit_unary(cf, "HARD_TANH",  o, a, n, "x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x)"); break;
        case UOP_SOFTPLUS:   aot_emit_unary(cf, "SOFTPLUS",   o, a, n, "_cml_softplus(x)"); break;
        case UOP_SOFTSIGN:   aot_emit_unary(cf, "SOFTSIGN",   o, a, n, "x / (1.0f + fabsf(x))"); break;
        case UOP_LOGSIGMOID: aot_emit_unary(cf, "LOGSIGMOID", o, a, n, "-_cml_softplus(-x)"); break;
        case UOP_QUICK_GELU: aot_emit_unary(cf, "QUICK_GELU", o, a, n, "x * _cml_sigmoid(1.702f * x)"); break;
        case UOP_MISH:       aot_emit_unary(cf, "MISH",       o, a, n, "x * tanhf(_cml_softplus(x))"); break;
        case UOP_SILU:       aot_emit_unary(cf, "SILU",       o, a, n, "x * _cml_sigmoid(x)"); break;
        case UOP_HARDSWISH:  aot_emit_unary(cf, "HARDSWISH",  o, a, n, "x < -3.0f ? 0.0f : (x > 3.0f ? x : x * (x + 3.0f) / 6.0f)"); break;
        case UOP_SELU:       aot_emit_unary(cf, "SELU",       o, a, n, "1.05070098f * (x > 0.0f ? x : 1.67326324f * (expf(x) - 1.0f))"); break;

        /* ---- Param unary ---- */
        case UOP_CLAMP: {
            ClampParams* p = (ClampParams*)node->params;
            float lo = p ? p->min_val : 0.0f, hi = p ? p->max_val : 0.0f;
            char expr[128];
            snprintf(expr, sizeof(expr), "x < %.9gf ? %.9gf : (x > %.9gf ? %.9gf : x)", lo, lo, hi, hi);
            aot_emit_unary(cf, "CLAMP", o, a, n, expr);
            break;
        }
        case UOP_ELU: {
            /* alpha stored in ClampParams.min_val (see uop_elu) */
            ClampParams* p = (ClampParams*)node->params;
            float alpha = p ? p->min_val : 1.0f;
            char expr[128];
            snprintf(expr, sizeof(expr), "x > 0.0f ? x : %.9gf * (expf(x) - 1.0f)", alpha);
            aot_emit_unary(cf, "ELU", o, a, n, expr);
            break;
        }
        case UOP_CELU: {
            /* alpha stored in ClampParams.min_val (see uop_celu) */
            ClampParams* p = (ClampParams*)node->params;
            float alpha = p ? p->min_val : 1.0f;
            char expr[160];
            snprintf(expr, sizeof(expr),
                     "fmaxf(0.0f, x) + fminf(0.0f, %.9gf * (expf(x / %.9gf) - 1.0f))", alpha, alpha);
            aot_emit_unary(cf, "CELU", o, a, n, expr);
            break;
        }

        /* ---- Zero-input creation ---- */
        case UOP_FILL: {
            FillParams* p = (FillParams*)node->params;
            float v = p ? p->value : 0.0f;
            fprintf(cf, "    /* FILL */\n");
            fprintf(cf, "    for (int64_t i = 0; i < %lld; i++) %s[i] = %.9gf;\n",
                    (long long)n, o, v);
            break;
        }
        case UOP_CONST: {
            ConstParams* p = (ConstParams*)node->params;
            if (!p || !p->data) {
                LOG_ERROR("AOT: CONST node has no data");
                emit_ok = false;
                break;
            }
            int64_t cn = (int64_t)(p->data_size / sizeof(float));
            const float* cd = (const float*)p->data;
            int64_t copy_n = cn < n ? cn : n;
            fprintf(cf, "    /* CONST */\n");
            fprintf(cf, "    {\n        static const float _c[%lld] = {", (long long)(cn > 0 ? cn : 1));
            for (int64_t i = 0; i < cn; i++)
                fprintf(cf, "%s%.9gf", i ? ", " : "", cd[i]);
            fprintf(cf, "};\n        memcpy(%s, _c, %lld * sizeof(float));\n    }\n",
                    o, (long long)copy_n);
            break;
        }

        /* ---- Movement (contiguous view => copy) ---- */
        case UOP_RESHAPE:
        case UOP_FLATTEN:
        case UOP_UNFLATTEN:
            if (!in0) { LOG_ERROR("AOT: movement op missing input"); emit_ok = false; break; }
            fprintf(cf, "    /* %s (copy) */\n",
                    node->type == UOP_RESHAPE ? "RESHAPE" :
                    node->type == UOP_FLATTEN ? "FLATTEN" : "UNFLATTEN");
            fprintf(cf, "    memcpy(%s, %s, %lld * sizeof(float));\n",
                    o, a, (long long)(na < n ? na : n));
            break;

        case UOP_EXPAND:
            if (!in0) { LOG_ERROR("AOT: EXPAND missing input"); emit_ok = false; break; }
            if (na == 1) {
                fprintf(cf, "    /* EXPAND (scalar) */\n");
                fprintf(cf, "    for (int64_t i = 0; i < %lld; i++) %s[i] = %s[0];\n",
                        (long long)n, o, a);
            } else if (na == n) {
                fprintf(cf, "    /* EXPAND (copy) */\n");
                fprintf(cf, "    memcpy(%s, %s, %lld * sizeof(float));\n", o, a, (long long)n);
            } else {
                LOG_ERROR("AOT: EXPAND with non-trivial broadcast not supported");
                emit_ok = false;
            }
            break;

        case UOP_PERMUTE: {
            PermuteParams* p = (PermuteParams*)node->params;
            if (!in0 || !p || !p->perm || p->num_dims != in0->ndim || in0->ndim < 1) {
                LOG_ERROR("AOT: PERMUTE with unsupported params");
                emit_ok = false;
                break;
            }
            int nd = in0->ndim;
            if (nd > 16) { LOG_ERROR("AOT: PERMUTE ndim > 16 unsupported"); emit_ok = false; break; }
            /* Contiguous input strides; output shape/strides follow perm. */
            int64_t istr[16], osh[16], ostr[16];
            istr[nd - 1] = 1;
            for (int d = nd - 2; d >= 0; d--) istr[d] = istr[d + 1] * in0->shape[d + 1];
            for (int d = 0; d < nd; d++) osh[d] = in0->shape[p->perm[d]];
            ostr[nd - 1] = 1;
            for (int d = nd - 2; d >= 0; d--) ostr[d] = ostr[d + 1] * osh[d + 1];

            fprintf(cf, "    /* PERMUTE */\n");
            fprintf(cf, "    for (int64_t idx = 0; idx < %lld; idx++) {\n", (long long)n);
            fprintf(cf, "        int64_t rem = idx, in_off = 0;\n");
            for (int d = 0; d < nd; d++) {
                int64_t in_axis_stride = istr[p->perm[d]];
                fprintf(cf, "        { int64_t c = rem / %lld; rem -= c * %lld; in_off += c * %lld; }\n",
                        (long long)ostr[d], (long long)ostr[d], (long long)in_axis_stride);
            }
            fprintf(cf, "        %s[idx] = %s[in_off];\n", o, a);
            fprintf(cf, "    }\n");
            break;
        }

        /* ---- Reductions ---- */
        case UOP_SUM: case UOP_MEAN: case UOP_PROD:
        case UOP_MAX_REDUCE: case UOP_MIN_REDUCE: {
            if (!in0) { LOG_ERROR("AOT: reduction missing input"); emit_ok = false; break; }
            ReduceParams* p = (ReduceParams*)node->params;
            int nd = in0->ndim;
            int64_t outer = 1, dim = na, inner = 1;
            bool full = (!p || p->num_dims == 0 || p->num_dims == nd);
            if (!full && p->num_dims == 1) {
                int d = p->dims[0];
                if (d < 0) d += nd;
                if (d < 0 || d >= nd) { LOG_ERROR("AOT: bad reduce dim"); emit_ok = false; break; }
                for (int i = 0; i < d; i++) outer *= in0->shape[i];
                dim = in0->shape[d];
                for (int i = d + 1; i < nd; i++) inner *= in0->shape[i];
            } else if (!full) {
                LOG_ERROR("AOT: multi-dim partial reduction not supported");
                emit_ok = false;
                break;
            }
            const char *name, *init, *acc, *fin = "acc";
            switch (node->type) {
            case UOP_SUM:        name = "SUM";        init = "0.0f";      acc = "acc + v"; break;
            case UOP_MEAN:       name = "MEAN";       init = "0.0f";      acc = "acc + v"; break;
            case UOP_PROD:       name = "PROD";       init = "1.0f";      acc = "acc * v"; break;
            case UOP_MAX_REDUCE: name = "MAX_REDUCE"; init = "-INFINITY"; acc = "v > acc ? v : acc"; break;
            default:             name = "MIN_REDUCE"; init = "INFINITY";  acc = "v < acc ? v : acc"; break;
            }
            if (node->type == UOP_MEAN) {
                char finbuf[64];
                snprintf(finbuf, sizeof(finbuf), "acc / %.1ff", (double)dim);
                aot_emit_reduce(cf, name, o, a, outer, dim, inner, init, acc, finbuf);
            } else {
                aot_emit_reduce(cf, name, o, a, outer, dim, inner, init, acc, fin);
            }
            break;
        }

        /* ---- Matmul / Linear ---- */
        case UOP_MATMUL: {
            if (!in0 || !in1) { LOG_ERROR("AOT: MATMUL missing inputs"); emit_ok = false; break; }
            int64_t M = 1, K = 1, N = 1;
            if (in0->ndim >= 2) { M = in0->shape[in0->ndim - 2]; K = in0->shape[in0->ndim - 1]; }
            else if (in0->ndim == 1) { M = 1; K = in0->shape[0]; }
            if (in1->ndim >= 2) { N = in1->shape[in1->ndim - 1]; }
            else if (in1->ndim == 1) { N = 1; }
            fprintf(cf, "    /* MATMUL */\n    {\n");
            fprintf(cf, "        int64_t M = %lld, K = %lld, N = %lld;\n",
                    (long long)M, (long long)K, (long long)N);
            fprintf(cf, "        for (int64_t m = 0; m < M; m++)\n");
            fprintf(cf, "            for (int64_t nn = 0; nn < N; nn++) {\n");
            fprintf(cf, "                float acc = 0.0f;\n");
            fprintf(cf, "                for (int64_t k = 0; k < K; k++)\n");
            fprintf(cf, "                    acc += %s[m * K + k] * %s[k * N + nn];\n", a, b);
            fprintf(cf, "                %s[m * N + nn] = acc;\n", o);
            fprintf(cf, "            }\n    }\n");
            break;
        }
        case UOP_LINEAR: {
            /* inputs: [input, weight(N,K), bias(N)?]; out = input @ weight^T + bias */
            if (!in0 || !in1) { LOG_ERROR("AOT: LINEAR missing inputs"); emit_ok = false; break; }
            Tensor* bias = (node->num_inputs > 2 && node->inputs) ? node->inputs[2] : NULL;
            int64_t N = in1->shape[0];
            int64_t K = in1->ndim >= 2 ? in1->shape[1] : in1->shape[0];
            int64_t M = K > 0 ? (int64_t)in0->numel / K : 0;
            char bexpr[64] = "";
            if (bias) aot_ptr_expr(&map, bias, bexpr, sizeof(bexpr));
            fprintf(cf, "    /* LINEAR */\n    {\n");
            fprintf(cf, "        int64_t M = %lld, K = %lld, N = %lld;\n",
                    (long long)M, (long long)K, (long long)N);
            fprintf(cf, "        for (int64_t m = 0; m < M; m++)\n");
            fprintf(cf, "            for (int64_t nn = 0; nn < N; nn++) {\n");
            fprintf(cf, "                float acc = 0.0f;\n");
            fprintf(cf, "                for (int64_t k = 0; k < K; k++)\n");
            fprintf(cf, "                    acc += %s[m * K + k] * %s[nn * K + k];\n", a, b);
            if (bias)
                fprintf(cf, "                acc += %s[nn];\n", bexpr);
            fprintf(cf, "                %s[m * N + nn] = acc;\n", o);
            fprintf(cf, "            }\n    }\n");
            break;
        }

        default:
            LOG_ERROR("AOT: unsupported UOp %d has no C-codegen path", (int)node->type);
            emit_ok = false;
            break;
        }
    }

    if (num_intermediates > 0)
        fprintf(cf, "_cml_cleanup:\n");
    for (int i = 0; i < map.count; i++)
        if (map.items[i].kind == AOT_KIND_INTERMEDIATE)
            fprintf(cf, "    free(t%d);\n", map.items[i].index);
    fprintf(cf, "}\n");
    fclose(cf);
    free(map.items);

    if (!emit_ok) {
        LOG_ERROR("AOT: code generation failed, aborting");
        remove(tmp_c_path);
        return -1;
    }

    if (opts.format == AOT_FORMAT_LLVM_IR) {
        /* For LLVM_IR format, just copy the generated C source to output_path */
        FILE* src = fopen(tmp_c_path, "r");
        if (!src) {
            LOG_ERROR("Failed to re-open temp C file");
            remove(tmp_c_path);
            return -1;
        }
        FILE* dst = fopen(output_path, "w");
        if (!dst) {
            LOG_ERROR("Failed to open output: %s", output_path);
            fclose(src);
            remove(tmp_c_path);
            return -1;
        }
        char buf[4096];
        size_t nread;
        while ((nread = fread(buf, 1, sizeof(buf), src)) > 0)
            fwrite(buf, 1, nread, dst);
        fclose(src);
        fclose(dst);
        remove(tmp_c_path);
        LOG_INFO("AOT: wrote generated C source to %s", output_path);
        return 0;
    }

    if (opts.format == AOT_FORMAT_OBJECT) {
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "cc -O2 -c -o %s %s", output_path, tmp_c_path);
        FILE* proc = popen(cmd, "r");
        if (!proc) {
            LOG_ERROR("Failed to invoke compiler: %s", cmd);
            remove(tmp_c_path);
            return -1;
        }
        char proc_buf[256];
        while (fgets(proc_buf, sizeof(proc_buf), proc)) {
            LOG_INFO("cc: %s", proc_buf);
        }
        int status = pclose(proc);
        remove(tmp_c_path);
        if (status != 0) {
            LOG_ERROR("Compiler failed with status %d", status);
            return -1;
        }
        LOG_INFO("AOT: compiled object file %s", output_path);
        return 0;
    }

    if (opts.format == AOT_FORMAT_SHARED_LIB) {
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "cc -O2 -fPIC -shared -o %s %s -lm",
                 output_path, tmp_c_path);
        FILE* proc = popen(cmd, "r");
        if (!proc) {
            LOG_ERROR("Failed to invoke compiler: %s", cmd);
            remove(tmp_c_path);
            return -1;
        }
        char proc_buf[256];
        while (fgets(proc_buf, sizeof(proc_buf), proc)) {
            LOG_INFO("cc: %s", proc_buf);
        }
        int status = pclose(proc);
        remove(tmp_c_path);
        if (status != 0) {
            LOG_ERROR("Compiler failed with status %d", status);
            return -1;
        }
        LOG_INFO("AOT: compiled shared library %s", output_path);
        return 0;
    }

    if (opts.format == AOT_FORMAT_STATIC_LIB) {
        /* First compile to object file */
        char tmp_o_path[512];
        snprintf(tmp_o_path, sizeof(tmp_o_path), "/tmp/cml_aot_%d.o", (int)getpid());

        char cmd[2048];
        snprintf(cmd, sizeof(cmd), "cc -O2 -c -o %s %s", tmp_o_path, tmp_c_path);
        FILE* proc = popen(cmd, "r");
        if (!proc) {
            LOG_ERROR("Failed to invoke compiler: %s", cmd);
            remove(tmp_c_path);
            return -1;
        }
        char proc_buf[256];
        while (fgets(proc_buf, sizeof(proc_buf), proc)) {
            LOG_INFO("cc: %s", proc_buf);
        }
        int status = pclose(proc);
        remove(tmp_c_path);
        if (status != 0) {
            LOG_ERROR("Compiler failed with status %d", status);
            return -1;
        }

        /* Then archive into static lib */
        if (!aot_validate_path(tmp_o_path)) {
            LOG_ERROR("Unsafe temp object path");
            remove(tmp_o_path);
            return -1;
        }
        snprintf(cmd, sizeof(cmd), "ar rcs %s %s", output_path, tmp_o_path);
        proc = popen(cmd, "r");
        if (!proc) {
            LOG_ERROR("Failed to invoke ar: %s", cmd);
            remove(tmp_o_path);
            return -1;
        }
        while (fgets(proc_buf, sizeof(proc_buf), proc)) {
            LOG_INFO("ar: %s", proc_buf);
        }
        status = pclose(proc);
        remove(tmp_o_path);
        if (status != 0) {
            LOG_ERROR("ar failed with status %d", status);
            return -1;
        }
        LOG_INFO("AOT: created static library %s", output_path);
        return 0;
    }

    /* Unknown format */
    remove(tmp_c_path);
    LOG_ERROR("Unknown AOT format: %d", (int)opts.format);
    return -1;

#else
    (void)ir; (void)output_path; (void)options;
    LOG_ERROR("AOT compilation requires LLVM backend support");
    return -1;
#endif
}

int cml_aot_compile_module(struct Module* module, Tensor* sample_input,
                           const char* output_path, const AOTCompileOptions* options) {
    if (!module || !sample_input || !output_path) {
        LOG_ERROR("Invalid arguments to cml_aot_compile_module");
        return -1;
    }

    LOG_INFO("AOT compiling module '%s'", module->name ? module->name : "unnamed");

    /* Trace the forward pass to capture IR */
    cml_ir_reset_global_context();
    Tensor* output = module_forward(module, sample_input);
    if (!output) {
        LOG_ERROR("Forward pass failed during AOT tracing");
        return -1;
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        LOG_ERROR("No IR captured during forward pass");
        return -1;
    }

    return cml_aot_compile(ir, output_path, options);
}

CMLAOTModel* cml_aot_load(const char* path) {
    if (!path) {
        LOG_ERROR("NULL path for AOT model load");
        return NULL;
    }

    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        LOG_ERROR("Failed to load AOT model: %s", dlerror());
        return NULL;
    }

    /* Look up the entry function */
    void* forward_fn = dlsym(handle, "cml_model_forward");
    if (!forward_fn) {
        /* Try alternate name */
        forward_fn = dlsym(handle, "main");
    }

    if (!forward_fn) {
        LOG_ERROR("No entry function found in AOT model");
        dlclose(handle);
        return NULL;
    }

    CMLAOTModel* model = calloc(1, sizeof(CMLAOTModel));
    if (!model) {
        dlclose(handle);
        return NULL;
    }

    model->handle = handle;
    model->forward_fn = forward_fn;
    model->path = strdup(path);

    LOG_INFO("AOT model loaded from: %s", path);
    return model;
}

int cml_aot_execute(CMLAOTModel* model, Tensor** inputs, int num_inputs,
                    Tensor** outputs, int num_outputs) {
    if (!model || !model->forward_fn || !inputs || !outputs) {
        LOG_ERROR("Invalid arguments to cml_aot_execute");
        return -1;
    }

    /* Memref descriptor matching the generated code's layout. */
    typedef struct {
        float* allocated;
        float* aligned;
        int64_t offset;
        int64_t sizes[8];
        int64_t strides[8];
    } MemRef;

    int total = num_inputs + num_outputs;
    MemRef* descs = calloc((size_t)total, sizeof(MemRef));
    MemRef** in_ptrs = calloc((size_t)(num_inputs > 0 ? num_inputs : 1), sizeof(MemRef*));
    MemRef** out_ptrs = calloc((size_t)(num_outputs > 0 ? num_outputs : 1), sizeof(MemRef*));
    if (!descs || !in_ptrs || !out_ptrs) {
        free(descs); free(in_ptrs); free(out_ptrs);
        return -1;
    }

    for (int i = 0; i < total; i++) {
        Tensor* t = i < num_inputs ? inputs[i] : outputs[i - num_inputs];
        /* Allocate output data lazily if the caller left it NULL. */
        if (t && !t->data && t->numel > 0) {
            t->data = calloc(t->numel, sizeof(float));
            t->owns_data = true;
        }
        if (!t || !t->data) {
            free(descs); free(in_ptrs); free(out_ptrs);
            return -1;
        }
        descs[i].allocated = (float*)t->data;
        descs[i].aligned = (float*)t->data;
        descs[i].offset = 0;
        descs[i].sizes[0] = (int64_t)t->numel;
        descs[i].strides[0] = 1;
        if (i < num_inputs)
            in_ptrs[i] = &descs[i];
        else
            out_ptrs[i - num_inputs] = &descs[i];
    }

    typedef void (*ForwardFn)(MemRef**, MemRef**);
    ForwardFn fn = (ForwardFn)model->forward_fn;
    fn(in_ptrs, out_ptrs);

    free(descs);
    free(in_ptrs);
    free(out_ptrs);
    return 0;
}

void cml_aot_free(CMLAOTModel* model) {
    if (!model)
        return;

    if (model->handle)
        dlclose(model->handle);

    if (model->input_shapes) {
        for (int i = 0; i < model->num_inputs; i++)
            free(model->input_shapes[i]);
        free(model->input_shapes);
    }
    if (model->output_shapes) {
        for (int i = 0; i < model->num_outputs; i++)
            free(model->output_shapes[i]);
        free(model->output_shapes);
    }
    free(model->input_ndims);
    free(model->output_ndims);
    free((char*)model->path);
    free(model);
}

int cml_aot_generate_header(CMLGraph_t ir, const char* header_path, const char* function_name) {
    if (!header_path) {
        LOG_ERROR("NULL header path");
        return -1;
    }

    const char* fname = function_name ? function_name : "cml_model_forward";

    FILE* f = fopen(header_path, "w");
    if (!f) {
        LOG_ERROR("Failed to create header: %s", header_path);
        return -1;
    }

    /* Count real graph inputs (consumed tensors that are never produced). */
    int num_inputs = 0, num_outputs = 1;
    if (ir) {
        int produced_cap = 0, produced_cnt = 0;
        Tensor** produced = NULL;
        for (struct IRNode* nd = ir->head; nd; nd = nd->next) {
            if (!nd->output) continue;
            if (produced_cnt == produced_cap) {
                produced_cap = produced_cap ? produced_cap * 2 : 16;
                Tensor** np = realloc(produced, (size_t)produced_cap * sizeof(Tensor*));
                if (!np) break;
                produced = np;
            }
            produced[produced_cnt++] = nd->output;
        }
        int seen_cap = 0, seen_cnt = 0;
        Tensor** seen = NULL;
        for (struct IRNode* nd = ir->head; nd; nd = nd->next) {
            for (int k = 0; k < nd->num_inputs; k++) {
                Tensor* in = nd->inputs ? nd->inputs[k] : NULL;
                if (!in) continue;
                bool is_produced = false;
                for (int i = 0; i < produced_cnt; i++)
                    if (produced[i] == in) { is_produced = true; break; }
                if (is_produced) continue;
                bool already = false;
                for (int i = 0; i < seen_cnt; i++)
                    if (seen[i] == in) { already = true; break; }
                if (already) continue;
                if (seen_cnt == seen_cap) {
                    seen_cap = seen_cap ? seen_cap * 2 : 16;
                    Tensor** ns = realloc(seen, (size_t)seen_cap * sizeof(Tensor*));
                    if (!ns) break;
                    seen = ns;
                }
                seen[seen_cnt++] = in;
                num_inputs++;
            }
        }
        free(produced);
        free(seen);
    }

    fprintf(f, "/* Auto-generated CML AOT model header */\n");
    fprintf(f, "#ifndef CML_AOT_MODEL_H\n");
    fprintf(f, "#define CML_AOT_MODEL_H\n\n");
    fprintf(f, "#include <stdint.h>\n\n");
    fprintf(f, "#ifdef __cplusplus\n");
    fprintf(f, "extern \"C\" {\n");
    fprintf(f, "#endif\n\n");

    /* Memref descriptor type (matches the generated code / cml_aot_execute). */
    fprintf(f, "typedef struct {\n");
    fprintf(f, "    float* allocated;\n");
    fprintf(f, "    float* aligned;\n");
    fprintf(f, "    int64_t offset;\n");
    fprintf(f, "    int64_t sizes[8];\n");
    fprintf(f, "    int64_t strides[8];\n");
    fprintf(f, "} CMLMemRef;\n\n");

    fprintf(f, "/* Forward pass. inputs[] has %d entries, outputs[] has %d entry. */\n",
            num_inputs, num_outputs);
    fprintf(f, "void %s(CMLMemRef** inputs, CMLMemRef** outputs);\n\n", fname);

    fprintf(f, "#ifdef __cplusplus\n");
    fprintf(f, "}\n");
    fprintf(f, "#endif\n\n");
    fprintf(f, "#endif /* CML_AOT_MODEL_H */\n");

    fclose(f);
    LOG_INFO("Generated AOT header: %s", header_path);
    return 0;
}
