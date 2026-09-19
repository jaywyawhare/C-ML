/*
 * ONNX model export.
 *
 * Serializes a CML IR graph to the ONNX protobuf format (opset 11), reusing
 * the CMLONNXModel-era conventions the importer (onnx_ops.c) parses. The
 * exporter and importer are independent implementations, so a model is
 * validated by exporting it, parsing it back with cml_onnx_load_buffer, and
 * executing it with cml_onnx_run -- agreement with the original eager result
 * exercises both directions.
 *
 * Ops with no ONNX equivalent fail loudly with the UOp name rather than
 * emitting garbage.
 */
#include "core/onnx.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── minimal protobuf writer ──────────────────────────────────────────────── */

typedef struct {
    uint8_t* data;
    size_t len;
    size_t cap;
} PBBuf;

static void pb_init(PBBuf* b) {
    b->data = NULL;
    b->len  = 0;
    b->cap  = 0;
}

static void pb_free(PBBuf* b) {
    free(b->data);
    b->data = NULL;
    b->len  = 0;
    b->cap  = 0;
}

static bool pb_reserve(PBBuf* b, size_t extra) {
    if (b->len + extra <= b->cap)
        return true;
    size_t ncap = b->cap ? b->cap : 256;
    while (ncap < b->len + extra)
        ncap *= 2;
    uint8_t* nd = (uint8_t*)realloc(b->data, ncap);
    if (!nd)
        return false;
    b->data = nd;
    b->cap  = ncap;
    return true;
}

static bool pb_put(PBBuf* b, const void* p, size_t n) {
    if (!pb_reserve(b, n))
        return false;
    if (n && p)
        memcpy(b->data + b->len, p, n); /* memcpy(_, NULL, 0) is UB */
    b->len += n;
    return true;
}

static bool pb_varint(PBBuf* b, uint64_t v) {
    uint8_t tmp[10];
    int n = 0;
    do {
        uint8_t byte = (uint8_t)(v & 0x7F);
        v >>= 7;
        if (v)
            byte |= 0x80;
        tmp[n++] = byte;
    } while (v);
    return pb_put(b, tmp, (size_t)n);
}

static bool pb_tag(PBBuf* b, int field, int wt) {
    return pb_varint(b, ((uint64_t)field << 3) | (uint64_t)wt);
}

#define PB_WT_VARINT 0
#define PB_WT_LEN 2
#define PB_WT_FIXED32 5

static bool pb_field_varint(PBBuf* b, int field, uint64_t v) {
    return pb_tag(b, field, PB_WT_VARINT) && pb_varint(b, v);
}

static bool pb_field_float(PBBuf* b, int field, float f) {
    return pb_tag(b, field, PB_WT_FIXED32) && pb_put(b, &f, 4);
}

static bool pb_field_len(PBBuf* b, int field, const void* p, size_t n) {
    return pb_tag(b, field, PB_WT_LEN) && pb_varint(b, n) && pb_put(b, p, n);
}

static bool pb_field_buf(PBBuf* b, int field, const PBBuf* child) {
    return pb_field_len(b, field, child->data, child->len);
}

static bool pb_field_str(PBBuf* b, int field, const char* s) {
    return pb_field_len(b, field, s, strlen(s));
}

static bool pb_field_packed_int64(PBBuf* b, int field, const int64_t* v, int n) {
    PBBuf payload;
    pb_init(&payload);
    bool ok = true;
    for (int i = 0; i < n; i++)
        ok = ok && pb_varint(&payload, (uint64_t)v[i]);
    if (ok)
        ok = pb_field_buf(b, field, &payload);
    pb_free(&payload);
    return ok;
}

/* ── ONNX constants ───────────────────────────────────────────────────────── */

#define ONNX_FLOAT 1
#define ONNX_UINT8 2
#define ONNX_INT8 3
#define ONNX_UINT16 4
#define ONNX_INT16 5
#define ONNX_INT32 6
#define ONNX_INT64 7
#define ONNX_BOOL 9
#define ONNX_FLOAT16 10
#define ONNX_DOUBLE 11
#define ONNX_BFLOAT16 16

static int onnx_elem_type(DType dt) {
    switch (dt) {
    case DTYPE_FLOAT32:
        return ONNX_FLOAT;
    case DTYPE_FLOAT64:
        return ONNX_DOUBLE;
    case DTYPE_INT8:
        return ONNX_INT8;
    case DTYPE_UINT8:
        return ONNX_UINT8;
    case DTYPE_INT16:
        return ONNX_INT16;
    case DTYPE_UINT16:
        return ONNX_UINT16;
    case DTYPE_INT32:
        return ONNX_INT32;
    case DTYPE_INT64:
        return ONNX_INT64;
    case DTYPE_BOOL:
        return ONNX_BOOL;
    case DTYPE_FLOAT16:
        return ONNX_FLOAT16;
    case DTYPE_BFLOAT16:
        return ONNX_BFLOAT16;
    default:
        return 0;
    }
}

#define OATTR_FLOAT 1
#define OATTR_INT 2
#define OATTR_STRING 3
#define OATTR_FLOATS 6
#define OATTR_INTS 7

/* ── export context ───────────────────────────────────────────────────────── */

typedef struct {
    Tensor* t;
    char name[64];
} TensorName;

typedef struct {
    PBBuf nodes;        /* repeated NodeProto      */
    PBBuf initializers; /* repeated TensorProto    */
    PBBuf vi_inputs;    /* ValueInfoProto, inputs  */
    PBBuf vi_outputs;   /* ValueInfoProto, outputs */

    TensorName* names;
    int num_names;
    int cap_names;

    Tensor** user_inputs;
    int num_user_inputs;

    int num_nodes_emitted;
    int num_initializers;
    char err[192];
} ExportCtx;

static const char* ctx_error(ExportCtx* c, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

static const char* ctx_error(ExportCtx* c, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->err, sizeof(c->err), fmt, ap);
    va_end(ap);
    return c->err;
}

/* Pointer-keyed registry: returns existing name for t, or registers `name`. */
static const char* intern_name(ExportCtx* c, Tensor* t, const char* name) {
    for (int i = 0; i < c->num_names; i++)
        if (c->names[i].t == t)
            return c->names[i].name;
    if (c->num_names == c->cap_names) {
        int nc         = c->cap_names ? c->cap_names * 2 : 32;
        TensorName* nn = (TensorName*)realloc(c->names, (size_t)nc * sizeof(TensorName));
        if (!nn) {
            ctx_error(c, "out of memory");
            return NULL;
        }
        c->names     = nn;
        c->cap_names = nc;
    }
    TensorName* e = &c->names[c->num_names++];
    e->t          = t;
    snprintf(e->name, sizeof(e->name), "%s", name ? name : "");
    return e->name;
}

static const char* lookup_name(ExportCtx* c, Tensor* t) {
    for (int i = 0; i < c->num_names; i++)
        if (c->names[i].t == t)
            return c->names[i].name;
    return NULL;
}

static bool is_user_input(ExportCtx* c, Tensor* t) {
    for (int i = 0; i < c->num_user_inputs; i++)
        if (c->user_inputs[i] == t)
            return true;
    return false;
}

/* Emit one TensorProto (initializer) for an eager tensor. */
static bool emit_initializer(ExportCtx* c, Tensor* t, const char* name) {
    if (!t->data && t->numel > 0) {
        tensor_ensure_executed(t);
        if (!t->data) {
            ctx_error(c, "tensor '%s' has no data", name);
            return false;
        }
    }
    int et = onnx_elem_type(t->dtype);
    if (!et) {
        ctx_error(c, "unsupported dtype %d for initializer", (int)t->dtype);
        return false;
    }
    PBBuf tp;
    pb_init(&tp);
    bool ok = true;
    if (t->ndim > 0) {
        int64_t dims[8];
        for (int i = 0; i < t->ndim && i < 8; i++)
            dims[i] = t->shape[i];
        ok = pb_field_packed_int64(&tp, 1, dims, t->ndim);
    }
    ok = ok && pb_field_varint(&tp, 2, (uint64_t)et);
    ok = ok && pb_field_str(&tp, 8, name);
    ok = ok && pb_field_len(&tp, 9, t->data, t->numel * cml_dtype_size(t->dtype));
    ok = ok && pb_field_buf(&c->initializers, 5, &tp);
    pb_free(&tp);
    if (ok)
        c->num_initializers++;
    else
        ctx_error(c, "failed to serialize initializer '%s'", name);
    return ok;
}

/* Name for an eager leaf tensor: graph input if user-listed, else initializer. */
static const char* leaf_name(ExportCtx* c, Tensor* t) {
    const char* existing = lookup_name(c, t);
    if (existing)
        return existing;

    char buf[64];
    if (is_user_input(c, t)) {
        for (int i = 0; i < c->num_user_inputs; i++) {
            if (c->user_inputs[i] == t) {
                snprintf(buf, sizeof(buf), "input_%d", i);
                return intern_name(c, t, buf);
            }
        }
    }
    snprintf(buf, sizeof(buf), "init_%d", c->num_initializers);
    const char* nm = intern_name(c, t, buf);
    if (!nm) {
        ctx_error(c, "out of memory");
        return NULL;
    }
    if (!emit_initializer(c, t, nm))
        return NULL;
    return nm;
}

/* ValueInfoProto{name=1, type=2 TypeProto{tensor_type=1{elem_type=1, shape=2}}}*/
static bool emit_value_info(PBBuf* dst, int field, Tensor* t, const char* name) {
    int et = onnx_elem_type(t->dtype);
    if (!et)
        return false;
    PBBuf shape, tt, vi;
    pb_init(&shape);
    pb_init(&tt);
    pb_init(&vi);
    bool ok = true;
    for (int i = 0; i < t->ndim; i++) {
        PBBuf dim;
        pb_init(&dim);
        ok = ok && pb_field_varint(&dim, 1, (uint64_t)t->shape[i]);
        ok = ok && pb_field_buf(&shape, 1, &dim);
        pb_free(&dim);
    }
    ok = ok && pb_field_varint(&tt, 1, (uint64_t)et);
    ok = ok && pb_field_buf(&tt, 2, &shape);
    ok = ok && pb_field_str(&vi, 1, name);
    ok = ok && pb_field_buf(&vi, 2, &tt);
    ok = ok && pb_field_buf(dst, field, &vi);
    pb_free(&shape);
    pb_free(&tt);
    pb_free(&vi);
    return ok;
}

/* AttributeProto emitters. Attributes are collected individually and each is
 * written to the NodeProto as its own tag-5 length-delimited field; bundling
 * several messages under one header would produce a single malformed
 * AttributeProto. */
#define ATTR_MAX 16

typedef struct {
    PBBuf msgs[ATTR_MAX];
    int n;
} AttrList;

static bool attr_push(AttrList* l, PBBuf* msg) {
    if (l->n >= ATTR_MAX) {
        pb_free(msg);
        return false;
    }
    l->msgs[l->n++] = *msg;
    pb_init(msg);
    return true;
}

static bool attr_int(AttrList* l, const char* name, int64_t v) {
    PBBuf a;
    pb_init(&a);
    bool ok = pb_field_str(&a, 1, name) && pb_field_varint(&a, 20, OATTR_INT) &&
              pb_field_varint(&a, 3, (uint64_t)v);
    return ok && attr_push(l, &a);
}

static bool attr_float_attr(AttrList* l, const char* name, float v) {
    PBBuf a;
    pb_init(&a);
    bool ok = pb_field_str(&a, 1, name) && pb_field_varint(&a, 20, OATTR_FLOAT) &&
              pb_field_float(&a, 2, v);
    return ok && attr_push(l, &a);
}

static bool attr_ints(AttrList* l, const char* name, const int64_t* v, int count) {
    PBBuf a;
    pb_init(&a);
    bool ok = pb_field_str(&a, 1, name) && pb_field_varint(&a, 20, OATTR_INTS) &&
              pb_field_packed_int64(&a, 8, v, count);
    return ok && attr_push(l, &a);
}

static bool emit_node_full(ExportCtx* c, const char* op_type, const char* const* ins, int nin,
                           const char* out_name, AttrList* attrs) {
    PBBuf nd;
    pb_init(&nd);
    bool ok = true;
    for (int i = 0; i < nin; i++)
        ok = ok && pb_field_str(&nd, 1, ins[i]); /* input (repeated) */
    ok = ok && pb_field_str(&nd, 2, out_name);   /* output */
    char name[80];
    snprintf(name, sizeof(name), "%s_%d", op_type, c->num_nodes_emitted);
    ok = ok && pb_field_str(&nd, 3, name);    /* name */
    ok = ok && pb_field_str(&nd, 4, op_type); /* op_type */
    if (attrs)
        for (int i = 0; i < attrs->n; i++)
            ok = ok && pb_field_buf(&nd, 5, &attrs->msgs[i]); /* attribute */
    ok = ok && pb_field_buf(&c->nodes, 1, &nd);
    pb_free(&nd);
    if (attrs)
        for (int i = 0; i < attrs->n; i++)
            pb_free(&attrs->msgs[i]);
    if (ok)
        c->num_nodes_emitted++;
    return ok;
}

static int norm_axis(int axis, int rank) {
    if (axis < 0)
        axis += rank;
    if (axis < 0)
        return 0;
    if (rank > 0 && axis >= rank)
        return rank - 1;
    return axis;
}

/* ── op mapping ───────────────────────────────────────────────────────────── */

static const char* map_uop(struct IRNode* n) {
    switch (n->type) {
    /* elementwise binary */
    case UOP_ADD:
        return "Add";
    case UOP_SUB:
        return "Sub";
    case UOP_MUL:
        return "Mul";
    case UOP_DIV:
        return "Div";
    case UOP_MAX:
        return "Max";
    case UOP_MINIMUM:
        return "Min";
    case UOP_POW:
        return "Pow";

    /* comparisons */
    case UOP_CMPLT:
        return "Less";
    case UOP_CMPGT:
        return "Greater";
    case UOP_CMPEQ:
        return "Equal";
    case UOP_CMPLE:
        return "LessOrEqual";
    case UOP_CMPGE:
        return "GreaterOrEqual";

    /* elementwise unary */
    case UOP_NEG:
        return "Neg";
    case UOP_EXP:
        return "Exp";
    case UOP_LOG:
        return "Log";
    case UOP_SQRT:
        return "Sqrt";
    case UOP_ABS:
        return "Abs";
    case UOP_SIN:
        return "Sin";
    case UOP_COS:
        return "Cos";
    case UOP_TANH:
        return "Tanh";
    case UOP_SIGMOID:
        return "Sigmoid";
    case UOP_SIGN:
        return "Sign";
    case UOP_FLOOR:
        return "Floor";
    case UOP_CEIL:
        return "Ceil";
    case UOP_ROUND:
        return "Round";
    case UOP_ERF:
        return "Erf";

    /* activations */
    case UOP_RELU:
        return "Relu";
    case UOP_SOFTPLUS:
        return "Softplus";
    case UOP_HARD_SIGMOID:
        return "HardSigmoid";
    case UOP_HARD_TANH:
        return "HardTanh";

    /* matmul / movement / misc */
    case UOP_MATMUL:
        return "MatMul";
    case UOP_RESHAPE:
        return "Reshape";
    case UOP_PERMUTE:
        return "Transpose";
    case UOP_EXPAND:
        return "Expand";
    case UOP_SLICE:
        return "Slice";
    case UOP_FLATTEN:
        return "Flatten";
    case UOP_CAT:
        return "Concat";
    case UOP_GATHER:
        return "Gather";
    case UOP_WHERE:
        return "Where";
    case UOP_CLAMP:
        return "Clip";

    /* conv/pool */
    case UOP_CONV2D:
        return "Conv";
    case UOP_MAXPOOL2D:
        return "MaxPool";
    case UOP_AVGPOOL2D:
        return "AveragePool";
    case UOP_CONV_TRANSPOSE2D:
        return "ConvTranspose";

    /* reductions */
    case UOP_SUM:
        return "ReduceSum";
    case UOP_MEAN:
        return "ReduceMean";
    case UOP_MAX_REDUCE:
        return "ReduceMax";
    case UOP_MIN_REDUCE:
        return "ReduceMin";
    case UOP_PROD:
        return "ReduceProd";

    default:
        return NULL;
    }
}

/* Attributes for ops carrying parameters. Returns false on error. */
static bool build_attrs(ExportCtx* c, struct IRNode* n, AttrList* attrs) {
    switch (n->type) {
    case UOP_PERMUTE: {
        PermuteParams* p = (PermuteParams*)n->params;
        if (!p || !p->perm) {
            ctx_error(c, "Permute missing params");
            return false;
        }
        int64_t perm[8];
        for (int i = 0; i < p->num_dims && i < 8; i++)
            perm[i] = p->perm[i];
        return attr_ints(attrs, "perm", perm, p->num_dims);
    }
    case UOP_CLAMP: {
        ClampParams* p = (ClampParams*)n->params;
        return attr_float_attr(attrs, "min", p ? p->min_val : -3.4e38f) &&
               attr_float_attr(attrs, "max", p ? p->max_val : 3.4e38f);
    }
    case UOP_FLATTEN: {
        FlattenParams* p = (FlattenParams*)n->params;
        int axis         = p ? p->start_dim : 1;
        if (p && p->end_dim != n->output_ndim - 1) {
            ctx_error(c, "Flatten over partial dim range not supported");
            return false;
        }
        return attr_int(attrs, "axis", norm_axis(axis, n->output_ndim + 1));
    }
    case UOP_CAT: {
        CatParams* p = (CatParams*)n->params;
        return attr_int(attrs, "axis", norm_axis(p ? p->dim : 0, n->output_ndim));
    }
    case UOP_GATHER: {
        GatherParams* p = (GatherParams*)n->params;
        int rank        = n->inputs[0] ? n->inputs[0]->ndim : 0;
        return attr_int(attrs, "axis", norm_axis(p ? p->dim : 0, rank));
    }
    case UOP_SUM:
    case UOP_MEAN:
    case UOP_MAX_REDUCE:
    case UOP_MIN_REDUCE:
    case UOP_PROD: {
        ReduceParams* p = (ReduceParams*)n->params;
        bool ok         = attr_int(attrs, "keepdims", (p && p->keepdim) ? 1 : 0);
        if (p && p->dims && p->num_dims > 0) {
            int64_t axes[8];
            int rank = n->inputs[0] ? n->inputs[0]->ndim : 0;
            for (int i = 0; i < p->num_dims && i < 8; i++)
                axes[i] = norm_axis(p->dims[i], rank);
            ok = attr_ints(attrs, "axes", axes, p->num_dims) && ok;
        }
        return ok;
    }
    case UOP_CONV2D: {
        Conv2DParams* p = (Conv2DParams*)n->params;
        if (!p) {
            ctx_error(c, "Conv2D missing params");
            return false;
        }
        return attr_ints(attrs, "kernel_shape", (int64_t[]){p->kernel_size[0], p->kernel_size[1]},
                         2) &&
               attr_ints(attrs, "strides", (int64_t[]){p->stride[0], p->stride[1]}, 2) &&
               attr_ints(attrs, "dilations", (int64_t[]){p->dilation[0], p->dilation[1]}, 2) &&
               attr_ints(attrs, "pads",
                         (int64_t[]){p->padding[0], p->padding[1], p->padding[0], p->padding[1]},
                         4) &&
               attr_int(attrs, "group", p->groups);
    }
    case UOP_MAXPOOL2D:
    case UOP_AVGPOOL2D: {
        Pool2DParams* p = (Pool2DParams*)n->params;
        if (!p) {
            ctx_error(c, "pool op missing params");
            return false;
        }
        bool ok =
            attr_ints(attrs, "kernel_shape", (int64_t[]){p->kernel_size[0], p->kernel_size[1]},
                      2) &&
            attr_ints(attrs, "strides", (int64_t[]){p->stride[0], p->stride[1]}, 2) &&
            attr_ints(attrs, "pads",
                      (int64_t[]){p->padding[0], p->padding[1], p->padding[0], p->padding[1]}, 4);
        if (n->type == UOP_MAXPOOL2D)
            ok = attr_int(attrs, "ceil_mode", p->ceil_mode ? 1 : 0) && ok;
        else
            ok = attr_int(attrs, "count_include_pad", p->count_include_pad ? 1 : 0) && ok;
        return ok;
    }
    default:
        return true;
    }
}

/* Shared scalar-1 f32 initializer for synthesized Div operands (RECIP). */
static const char* ensure_one_scalar(ExportCtx* c) {
    Tensor* existing = NULL;
    for (int i = 0; i < c->num_names; i++)
        if (strcmp(c->names[i].name, "cml_one_f32") == 0) {
            existing = c->names[i].t;
            break;
        }
    if (existing)
        return "cml_one_f32";

    TensorConfig cfg = {0};
    int s[1]         = {1};
    float v          = 1.0f;
    Tensor* one      = tensor_from_data(&v, s, 1, &cfg);
    if (!one || !one->data) {
        ctx_error(c, "failed to create scalar-1 constant");
        return NULL;
    }
    if (!emit_initializer(c, one, "cml_one_f32"))
        return NULL;
    return intern_name(c, one, "cml_one_f32");
}

/* Int64 shape-vector initializer (Reshape/Expand second operand). */
static const char* emit_shape_initializer(ExportCtx* c, const int* shape, int ndim,
                                          const char* name) {
    int64_t sv[8];
    for (int i = 0; i < ndim && i < 8; i++)
        sv[i] = shape[i];
    PBBuf tp;
    pb_init(&tp);
    bool ok = pb_field_packed_int64(&tp, 1, sv, ndim) && pb_field_varint(&tp, 2, ONNX_INT64) &&
              pb_field_str(&tp, 8, name) &&
              pb_field_len(&tp, 9, sv, (size_t)ndim * sizeof(int64_t)) &&
              pb_field_buf(&c->initializers, 5, &tp);
    pb_free(&tp);
    if (!ok) {
        ctx_error(c, "failed to serialize shape initializer");
        return NULL;
    }
    c->num_initializers++;
    return name;
}

/*
 * Emit a node whose operand set differs from the raw IR inputs.
 * Returns 0 = fully handled, -1 = error. `out_name` is the exported name of
 * this node's output tensor.
 */
static int export_special(ExportCtx* c, struct IRNode* n, const char* const* in_names, int nin,
                          const char* out_name) {
    switch (n->type) {
    case UOP_LINEAR: {
        /* Gemm(A, B, C?) with transB=1 */
        AttrList attrs = {0};
        bool ok        = attr_int(&attrs, "alpha", 1) && attr_int(&attrs, "beta", 1) &&
                  attr_int(&attrs, "transA", 0) && attr_int(&attrs, "transB", 1);
        ok = ok && emit_node_full(c, "Gemm", in_names, nin, out_name, &attrs);
        return ok ? 0 : -1;
    }
    case UOP_RECIP: {
        const char* one = ensure_one_scalar(c);
        if (!one)
            return -1;
        const char* ins[2] = {in_names[0], one};
        return emit_node_full(c, "Div", ins, 2, out_name, NULL) ? 0 : -1;
    }
    case UOP_SQUARE: {
        const char* ins[2] = {in_names[0], in_names[0]};
        return emit_node_full(c, "Mul", ins, 2, out_name, NULL) ? 0 : -1;
    }
    case UOP_RESHAPE:
    case UOP_EXPAND: {
        ShapeParams* p = (ShapeParams*)n->params;
        if (!p || !p->new_shape) {
            ctx_error(c, "%s missing shape params", n->type == UOP_RESHAPE ? "Reshape" : "Expand");
            return -1;
        }
        char buf[80];
        snprintf(buf, sizeof(buf), "%s_shape", out_name);
        const char* shp = emit_shape_initializer(c, p->new_shape, p->new_ndim, buf);
        if (!shp)
            return -1;
        const char* ins[2] = {in_names[0], shp};
        return emit_node_full(c, n->type == UOP_RESHAPE ? "Reshape" : "Expand", ins, 2, out_name,
                              NULL)
                   ? 0
                   : -1;
    }
    default:
        return 1; /* generic emission */
    }
}

int cml_onnx_export_graph(struct CMLGraph* ir_handle, Tensor** graph_inputs, int num_inputs,
                          Tensor** graph_outputs, int num_outputs, const char* filepath) {
    if (!ir_handle || !graph_outputs || num_outputs <= 0 || !filepath ||
        (num_inputs > 0 && !graph_inputs)) {
        LOG_ERROR("onnx_export: invalid arguments");
        return -1;
    }
    CMLGraph_t ir = (CMLGraph_t)ir_handle;

    ExportCtx c;
    memset(&c, 0, sizeof(c));
    pb_init(&c.nodes);
    pb_init(&c.initializers);
    pb_init(&c.vi_inputs);
    pb_init(&c.vi_outputs);
    c.user_inputs     = graph_inputs;
    c.num_user_inputs = num_inputs;

    int rc = -1;

    /* Walk head -> tail; producers precede consumers. */
    int idx          = 0;
    struct IRNode* n = ir->head;
    for (; n; n = n->next, idx++) {
        Tensor* out = n->output;
        if (!out)
            continue;

        /* Zero-input creation ops become plain initializers when their value
         * is already materialized (constants, fills, seeds). */
        if (n->num_inputs == 0) {
            if (out->data || (tensor_data_ptr(out) != NULL)) {
                char nm[40];
                snprintf(nm, sizeof(nm), "const_%d", idx);
                if (!emit_initializer(&c, out, nm))
                    goto done;
                if (!intern_name(&c, out, nm))
                    goto done;
                continue;
            }
            ctx_error(&c, "creation op %s has no materialized data to export",
                      uop_type_to_string(n->type));
            goto done;
        }

        /* Output name: prefer the node's own stable name, else generate. */
        const char* out_name = lookup_name(&c, out);
        if (!out_name) {
            char gen[48];
            if (out->ir_node && out->ir_node->output_name && out->ir_node->output_name[0])
                snprintf(gen, sizeof(gen), "%s", out->ir_node->output_name);
            else
                snprintf(gen, sizeof(gen), "n_%d", idx);
            out_name = intern_name(&c, out, gen);
        }
        if (!out_name)
            goto done;

        const char* op_type = map_uop(n);
        if (!op_type && n->type != UOP_LINEAR && n->type != UOP_RECIP && n->type != UOP_SQUARE) {
            ctx_error(&c, "UOp %s has no ONNX mapping", uop_type_to_string(n->type));
            goto done;
        }
        if (!op_type)
            op_type = "Add"; /* placeholder; special-cased below */

        /* Resolve operand names. Producers were emitted earlier (head->tail),
         * so their tensors are already registered under their exported names;
         * eager leaves become graph inputs or initializers. */
        const char* in_names[CML_ONNX_MAX_INPUTS];
        int nin = n->num_inputs < CML_ONNX_MAX_INPUTS ? n->num_inputs : CML_ONNX_MAX_INPUTS;
        for (int i = 0; i < nin; i++) {
            Tensor* t = n->inputs[i];
            if (!t) {
                ctx_error(&c, "node %s has NULL input %d", out_name, i);
                goto done;
            }
            if (t->ir_node) {
                const char* nm = lookup_name(&c, t);
                if (!nm) {
                    /* producer not yet visited (shared subgraph reordered?) --
                     * register under the producer's stable name now; its own
                     * emission will reuse the registration. */
                    nm = intern_name(&c, t, t->ir_node->output_name ? t->ir_node->output_name : "");
                    if (!nm || nm[0] == '\0') {
                        ctx_error(&c, "unnamed producer for input %d", i);
                        goto done;
                    }
                }
                in_names[i] = nm;
            } else {
                const char* nm = leaf_name(&c, t);
                if (!nm) {
                    LOG_ERROR("onnx_export: %s", c.err);
                    goto done;
                }
                in_names[i] = nm;
            }
        }

        int special = export_special(&c, n, in_names, nin, out_name);
        if (special < 0) {
            LOG_ERROR("onnx_export: %s", c.err);
            goto done;
        }
        if (special == 0)
            continue;

        AttrList attrs = {0};
        if (!build_attrs(&c, n, &attrs)) {
            LOG_ERROR("onnx_export: %s", c.err);
            goto done;
        }
        bool ok = emit_node_full(&c, op_type, in_names, nin, out_name, &attrs);
        if (!ok) {
            ctx_error(&c, "node serialization failed");
            goto done;
        }
    }

    /* Graph input value infos */
    for (int i = 0; i < num_inputs; i++) {
        Tensor* t = graph_inputs[i];
        if (!t) {
            ctx_error(&c, "NULL graph input %d", i);
            goto done;
        }
        char nm[32];
        snprintf(nm, sizeof(nm), "input_%d", i);
        /* Reuse an already-interned name if the tensor has one; otherwise
         * intern the input_%d name we just built. */
        const char* name = lookup_name(&c, t);
        if (!name) {
            if (!intern_name(&c, t, nm))
                goto done;
            name = nm;
        }
        if (!emit_value_info(&c.vi_inputs, 11, t, name)) {
            ctx_error(&c, "value info failed for %s", name);
            goto done;
        }
    }

    /* Graph output value infos */
    for (int i = 0; i < num_outputs; i++) {
        Tensor* t = graph_outputs[i];
        if (!t) {
            ctx_error(&c, "NULL graph output %d", i);
            goto done;
        }
        char on[32]; /* must outlive the if-block: nm may alias it below */
        const char* nm = lookup_name(&c, t);
        if (!nm) {
            /* eager leaf output: identity-copy from an initializer */
            const char* ln = leaf_name(&c, t);
            if (!ln)
                goto done;
            snprintf(on, sizeof(on), "output_%d", i);
            if (!emit_node_full(&c, "Identity", &ln, 1, on, NULL))
                goto done;
            nm = on;
        }
        if (!emit_value_info(&c.vi_outputs, 12, t, nm)) {
            ctx_error(&c, "value info failed for output %d", i);
            goto done;
        }
    }

    /* Assemble ModelProto */
    {
        PBBuf graph, opset, model;
        pb_init(&graph);
        pb_init(&opset);
        pb_init(&model);

        /* The accumulators already hold fully-tagged repeated elements
         * (each NodeProto/TensorProto/ValueInfoProto was written with its own
         * field tag on emission), so they are appended verbatim -- wrapping
         * them in pb_field_buf here would produce one giant sub-message. */
        bool ok = pb_put(&graph, c.nodes.data, c.nodes.len) &&               /* node */
                  pb_field_str(&graph, 2, "cml_graph") &&                    /* name */
                  pb_put(&graph, c.initializers.data, c.initializers.len) && /* initializer */
                  pb_put(&graph, c.vi_inputs.data, c.vi_inputs.len) &&       /* input */
                  pb_put(&graph, c.vi_outputs.data, c.vi_outputs.len);       /* output */

        /* OperatorSetIdProto{domain=1 "", version=2} */
        ok = ok && pb_field_str(&opset, 1, "") && pb_field_varint(&opset, 2, 11);

        ok = ok && pb_field_varint(&model, 1, 8) && /* ir_version */
             pb_field_str(&model, 2, "cml") &&      /* producer_name */
             pb_field_buf(&model, 7, &graph) &&     /* graph */
             pb_field_buf(&model, 8, &opset);       /* opset_import */

        if (ok) {
            FILE* f = fopen(filepath, "wb");
            if (!f) {
                LOG_ERROR("onnx_export: cannot open '%s' for writing", filepath);
            } else {
                size_t w = fwrite(model.data, 1, model.len, f);
                fclose(f);
                if (w == model.len) {
                    LOG_INFO("onnx_export: wrote %s (%zu bytes, %d nodes, "
                             "%d initializers)",
                             filepath, model.len, c.num_nodes_emitted, c.num_initializers);
                    rc = 0;
                } else {
                    LOG_ERROR("onnx_export: short write to '%s'", filepath);
                }
            }
        } else {
            LOG_ERROR("onnx_export: model assembly failed");
        }

        pb_free(&graph);
        pb_free(&opset);
        pb_free(&model);
    }

done:
    pb_free(&c.nodes);
    pb_free(&c.initializers);
    pb_free(&c.vi_inputs);
    pb_free(&c.vi_outputs);
    free(c.names);
    if (rc != 0)
        LOG_ERROR("onnx_export: export failed: %s", c.err[0] ? c.err : "unknown error");
    return rc;
}
