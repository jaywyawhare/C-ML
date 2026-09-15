/* Typed (non-float32) CPU execution.
 *
 * Every kernel here is written once against the element accessors in
 * tensor/dtype_access.h instead of a concrete C type. Two compute domains cover
 * all dtypes:
 *
 *   f64 domain  - exact for f32/f16/bf16/fp8 and for int8/int16/int32, since
 *                 double represents every value of those types exactly. int64
 *                 is exact to 2^53.
 *   i64 domain  - used by the bit-level ops, where a float representation is
 *                 meaningless. Exact for every integer width.
 *
 * Layout ops use neither: they move whole elements by size, which is exact for
 * every dtype including int64 and fp8.
 *
 * This costs a switch per element, so it is deliberately kept off the float32
 * hot path -- callers reach it only via cml_exec_needs_typed().
 */

#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ops/ir/execution_typed.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "tensor/dtype_access.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"
#include "ops/ir/cpu_lazy_materialize.h"
#include "backend/backend_buffer.h"

/* ---------------------------------------------------------------- helpers */

static bool cml_exec_out_dtype_is_role(UOpType t);
static size_t bcast_idx(const Tensor* inp, const Tensor* out, size_t flat_i);

static inline size_t bidx(size_t i, size_t n) { return n ? i % n : 0; }

static Tensor* in_at(struct IRNode* node, int i) {
    if (!node->inputs || node->num_inputs <= i || !node->inputs[i]) return NULL;
    return node->inputs[i];
}

/* Resolve a possibly-negative axis against `ndim`; -1 if out of range. */
static int axis_of(int dim, int ndim) {
    if (dim < 0) dim += ndim;
    return (dim < 0 || dim >= ndim) ? -1 : dim;
}

/* Split a tensor into (outer, count, inner) around `dim`: element (o,j,i) lives
 * at (o*count + j)*inner + i. Every axis-wise kernel below uses this. */
static void lanes_of(const Tensor* t, int dim, size_t* outer, size_t* count, size_t* inner) {
    size_t o = 1, n = 1;
    for (int d = 0; d < dim; d++) o *= (size_t)t->shape[d];
    for (int d = dim + 1; d < t->ndim; d++) n *= (size_t)t->shape[d];
    *outer = o;
    *count = (size_t)t->shape[dim];
    *inner = n;
}

/* Ops that already implement "handle the dtype or fail cleanly" inside the main
 * switch (cpu_matmul_generic / cpu_conv2d_generic). Routing them here would
 * shadow working code, so they stay on their own path. */
static bool handles_own_dtype(UOpType t) {
    return t == UOP_MATMUL || t == UOP_CONV2D;
}

bool cml_exec_needs_typed(struct IRNode* node, Tensor* out) {
    if (!node) return false;
    if (handles_own_dtype(node->type)) return false;
    /* Ops whose output dtype is fixed by their role (comparisons -> bool,
     * argmax/argsort -> index) are judged on their inputs; their f32 kernels
     * write the right thing for f32 inputs already. */
    for (int i = 0; i < node->num_inputs && node->inputs; i++) {
        Tensor* t = node->inputs[i];
        if (t && t->data && t->dtype != DTYPE_FLOAT32) return true;
    }
    return out && out->dtype != DTYPE_FLOAT32 && !cml_exec_out_dtype_is_role(node->type);
}

/* Output dtype is determined by the op's role rather than by its inputs, so an
 * f32-in/non-f32-out node is still a pure f32 kernel. */
static bool cml_exec_out_dtype_is_role(UOpType t) {
    switch (t) {
    case UOP_CMPLT: case UOP_CMPGT: case UOP_CMPLE:
    case UOP_CMPGE: case UOP_CMPEQ: case UOP_CMPNE:
    case UOP_ARGMAX: case UOP_ARGMIN: case UOP_ARGSORT:
    case UOP_NONZERO: case UOP_ALL: case UOP_ANY:
    case UOP_ISNAN: case UOP_ISINF: case UOP_ISFINITE:
    case UOP_LOGICAL_AND: case UOP_LOGICAL_OR: case UOP_LOGICAL_NOT:
        return true;
    default:
        return false;
    }
}

/* -------------------------------------------------- elementwise: f64 domain */

/* Every float-domain unary op, evaluated in double. Integer inputs promote to
 * double here and the result is stored back through the output dtype, which
 * matches the int->float promotion these ops imply mathematically. */
static int unary_f64(UOpType type, double x, double* r, struct IRNode* node) {
    ClampParams* cp = (ClampParams*)node->params;
    switch (type) {
    case UOP_NEG:      *r = -x;                       return 0;
    case UOP_ABS:      *r = fabs(x);                  return 0;
    case UOP_SQUARE:   *r = x * x;                    return 0;
    case UOP_EXP:      *r = exp(x);                   return 0;
    case UOP_LOG:      *r = log(x);                   return 0;
    case UOP_SQRT:     *r = sqrt(x);                  return 0;
    case UOP_RSQRT:    *r = 1.0 / sqrt(x);            return 0;
    case UOP_RECIP:    *r = 1.0 / x;                  return 0;
    case UOP_SIN:      *r = sin(x);                   return 0;
    case UOP_COS:      *r = cos(x);                   return 0;
    case UOP_TAN:      *r = tan(x);                   return 0;
    case UOP_TANH:     *r = tanh(x);                  return 0;
    case UOP_SIGMOID:  *r = 1.0 / (1.0 + exp(-x));    return 0;
    case UOP_FLOOR:    *r = floor(x);                 return 0;
    case UOP_CEIL:     *r = ceil(x);                  return 0;
    case UOP_ROUND:    *r = rint(x);                  return 0;  /* ties-to-even */
    case UOP_TRUNC:    *r = trunc(x);                 return 0;
    case UOP_LOG2:     *r = log2(x);                  return 0;
    case UOP_LOG10:    *r = log10(x);                 return 0;
    case UOP_EXP2:     *r = exp2(x);                  return 0;
    case UOP_ASIN:     *r = asin(x);                  return 0;
    case UOP_ACOS:     *r = acos(x);                  return 0;
    case UOP_ATAN:     *r = atan(x);                  return 0;
    case UOP_SINH:     *r = sinh(x);                  return 0;
    case UOP_COSH:     *r = cosh(x);                  return 0;
    case UOP_ASINH:    *r = asinh(x);                 return 0;
    case UOP_ACOSH:    *r = acosh(x);                 return 0;
    case UOP_ATANH:    *r = atanh(x);                 return 0;
    case UOP_ERF:      *r = erf(x);                   return 0;
    case UOP_ERFC:     *r = erfc(x);                  return 0;
    case UOP_SIGN:     *r = (x > 0.0) ? 1.0 : (x < 0.0 ? -1.0 : 0.0); return 0;
    case UOP_ISINF:    *r = isinf(x) ? 1.0 : 0.0;     return 0;
    case UOP_ISNAN:    *r = isnan(x) ? 1.0 : 0.0;     return 0;
    case UOP_ISFINITE: *r = isfinite(x) ? 1.0 : 0.0;  return 0;
    /* `x < 0 ? 0 : x`, not `x > 0 ? x : 0`: NaN fails both comparisons, so the
     * second form returns 0 and hides it. Same result for finite values. */
    case UOP_RELU:     *r = x < 0.0 ? 0.0 : x;        return 0;
    case UOP_RELU6:    *r = isnan(x) ? x : fmin(fmax(x, 0.0), 6.0);  return 0;
    case UOP_HARD_TANH:*r = isnan(x) ? x : fmin(fmax(x, -1.0), 1.0); return 0;
    case UOP_SOFTSIGN: *r = x / (1.0 + fabs(x));      return 0;
    case UOP_SILU:     *r = x / (1.0 + exp(-x));      return 0;
    case UOP_HARD_SIGMOID: *r = fmin(fmax((x + 3.0) / 6.0, 0.0), 1.0); return 0;
    case UOP_QUICK_GELU:   *r = x * (1.0 / (1.0 + exp(-1.702 * x)));   return 0;
    /* Stable softplus: max(x,0) + log1p(exp(-|x|)).
     * log(1 + y) loses most of y's significance when y is small -- softplus(-10)
     * came out with 4e-4 relative error, four orders worse than float epsilon.
     * log1p is exact there. */
    case UOP_SOFTPLUS:  *r = fmax(x, 0.0) + log1p(exp(-fabs(x)));  return 0;
    case UOP_LOGSIGMOID:*r = fmin(x, 0.0) - log1p(exp(-fabs(x)));  return 0;
    case UOP_MISH:      *r = x * tanh(fmax(x, 0.0) + log1p(exp(-fabs(x)))); return 0;
    case UOP_GELU: {
        double inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        *r = 0.5 * x * (1.0 + tanh(inner));
        return 0;
    }
    case UOP_HARDSWISH:
        *r = (x >= 3.0) ? x : (x <= -3.0 ? 0.0 : x * (x + 3.0) / 6.0);
        return 0;
    case UOP_SELU: {
        const double a = 1.6732632423543772, s = 1.0507009873554805;
        *r = s * (x > 0.0 ? x : a * (exp(x) - 1.0));
        return 0;
    }
    case UOP_ELU: {
        double alpha = cp ? (double)cp->min_val : 1.0;
        *r = x > 0.0 ? x : alpha * (exp(x) - 1.0);
        return 0;
    }
    case UOP_CELU: {
        /* max(0,x) + min(0, a*(exp(x/a)-1)) is the textbook form, but fmax/fmin
         * are IEEE maxNum/minNum: they return the *non*-NaN operand, so a NaN
         * input came back as 0. The branch below is the same function for every
         * alpha != 0 -- exactly one of the two terms is ever non-zero -- and it
         * propagates NaN for free, like the ELU line above. */
        double alpha = cp ? (double)cp->min_val : 1.0;
        *r = x > 0.0 ? x : alpha * (exp(x / alpha) - 1.0);
        return 0;
    }
    case UOP_LEAKY_RELU: {
        double slope = cp ? (double)cp->min_val : 0.01;
        *r = x > 0.0 ? x : slope * x;
        return 0;
    }
    case UOP_CLAMP: {
        double lo = cp ? (double)cp->min_val : -INFINITY;
        double hi = cp ? (double)cp->max_val : INFINITY;
        /* fmax/fmin return the non-NaN operand by definition (IEEE maxNum), so
         * clamping a NaN would silently yield the bound. */
        *r = isnan(x) ? x : fmin(fmax(x, lo), hi);
        return 0;
    }
    case UOP_LOGICAL_NOT: *r = (x == 0.0) ? 1.0 : 0.0; return 0;
    default: return -1;
    }
}

static int binary_f64(UOpType type, double x, double y, double* r) {
    switch (type) {
    case UOP_ADD:       *r = x + y;                 return 0;
    case UOP_SUB:       *r = x - y;                 return 0;
    case UOP_MUL:       *r = x * y;                 return 0;
    case UOP_DIV:       *r = x / y;                 return 0;
    case UOP_POW:       *r = pow(x, y);             return 0;
    /* NaN propagates, matching torch.maximum/minimum rather than IEEE maxNum. */
    case UOP_MAX:       *r = (isnan(x) || isnan(y)) ? (x + y) : (x > y ? x : y); return 0;
    case UOP_MINIMUM:   *r = (isnan(x) || isnan(y)) ? (x + y) : (x < y ? x : y); return 0;
    case UOP_MOD:       *r = fmod(x, y);            return 0;
    /* floor, not trunc: UOP_IDIV is documented as floor(a/b) and the f32 path
     * uses floorf. Truncating disagreed for negative operands (-1/2 gave 0
     * instead of -1), so the same graph answered differently per dtype. */
    case UOP_IDIV:      *r = (y != 0.0) ? floor(x / y) : 0.0; return 0;
    case UOP_COPYSIGN:  *r = copysign(x, y);        return 0;
    case UOP_LOGADDEXP: {
        double m = fmax(x, y);
        *r = isinf(m) ? m : m + log(exp(x - m) + exp(y - m));
        return 0;
    }
    case UOP_CMPLT: *r = (x <  y) ? 1.0 : 0.0; return 0;
    case UOP_CMPGT: *r = (x >  y) ? 1.0 : 0.0; return 0;
    case UOP_CMPLE: *r = (x <= y) ? 1.0 : 0.0; return 0;
    case UOP_CMPGE: *r = (x >= y) ? 1.0 : 0.0; return 0;
    case UOP_CMPEQ: *r = (x == y) ? 1.0 : 0.0; return 0;
    case UOP_CMPNE: *r = (x != y) ? 1.0 : 0.0; return 0;
    case UOP_LOGICAL_AND: *r = (x != 0.0 && y != 0.0) ? 1.0 : 0.0; return 0;
    case UOP_LOGICAL_OR:  *r = (x != 0.0 || y != 0.0) ? 1.0 : 0.0; return 0;
    default: return -1;
    }
}

/* -------------------------------------------------- elementwise: i64 domain */

/* Bit-level ops. These have no meaningful float representation, so they run on
 * the integer accessors and reject float dtypes at the dispatch site. */
static int binary_i64(UOpType type, int64_t x, int64_t y, int64_t* r) {
    switch (type) {
    case UOP_BITWISE_AND: *r = x & y; return 0;
    case UOP_BITWISE_OR:  *r = x | y; return 0;
    case UOP_BITWISE_XOR: *r = x ^ y; return 0;
    case UOP_LSHIFT:      *r = (y >= 0 && y < 64) ? (int64_t)((uint64_t)x << y) : 0; return 0;
    case UOP_RSHIFT:      *r = (y >= 0 && y < 64) ? (x >> y) : (x < 0 ? -1 : 0);    return 0;
    case UOP_IDIV: {                 /* floor division, see the f64 case */
        if (y == 0) { *r = 0; return 0; }
        int64_t q = x / y;
        if ((x % y != 0) && ((x < 0) != (y < 0))) q--;
        *r = q;
        return 0;
    }
    case UOP_MOD:         *r = (y != 0) ? x % y : 0; return 0;
    default: return -1;
    }
}

static int is_bitwise_op(UOpType t) {
    return t == UOP_BITWISE_AND || t == UOP_BITWISE_OR || t == UOP_BITWISE_XOR ||
           t == UOP_BITWISE_NOT || t == UOP_LSHIFT || t == UOP_RSHIFT;
}

/* Integer ops keep integer semantics on integer dtypes (C truncation for IDIV,
 * sign-following remainder for MOD); on float dtypes they fall to the f64
 * domain, where IDIV/MOD use trunc/fmod. Bitwise ops have no float form. */
static int prefers_i64(UOpType t, DType dt) {
    if (!cml_dtype_is_int(dt)) return 0;
    return is_bitwise_op(t) || t == UOP_IDIV || t == UOP_MOD;
}

/* ---------------------------------------------------------- dispatch: unary */

static int exec_unary(struct IRNode* node, Tensor* out) {
    Tensor* a = in_at(node, 0);
    if (!a || !a->data || !out->data) return -1;

    if (node->type == UOP_BITWISE_NOT) {
        if (!cml_dtype_is_int(a->dtype)) return -1;
        for (size_t i = 0; i < out->numel; i++)
            cml_store_i64(out->data, i, out->dtype,
                          ~cml_load_i64(a->data, bcast_idx(a, out, i), a->dtype));
        return 0;
    }

    double probe;
    if (unary_f64(node->type, 0.5, &probe, node) != 0) return -1;

    for (size_t i = 0; i < out->numel; i++) {
        double x = cml_load_f64(a->data, bcast_idx(a, out, i), a->dtype), r;
        if (unary_f64(node->type, x, &r, node) != 0) return -1;
        cml_store_f64(out->data, i, out->dtype, r);
    }
    return 0;
}

/* --------------------------------------------------------- dispatch: binary */

static int exec_binary(struct IRNode* node, Tensor* out) {
    Tensor* a = in_at(node, 0);
    Tensor* b = in_at(node, 1);
    if (!a || !b || !a->data || !b->data || !out->data) return -1;

    if (prefers_i64(node->type, a->dtype) && prefers_i64(node->type, b->dtype)) {
        int64_t probe;
        if (binary_i64(node->type, 4, 2, &probe) != 0) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            int64_t x = cml_load_i64(a->data, bcast_idx(a, out, i), a->dtype);
            int64_t y = cml_load_i64(b->data, bcast_idx(b, out, i), b->dtype);
            int64_t r;
            if (binary_i64(node->type, x, y, &r) != 0) return -1;
            cml_store_i64(out->data, i, out->dtype, r);
        }
        return 0;
    }
    if (is_bitwise_op(node->type)) return -1;  /* undefined on float dtypes */

    double probe;
    if (binary_f64(node->type, 1.0, 2.0, &probe) != 0) return -1;
    for (size_t i = 0; i < out->numel; i++) {
        double x = cml_load_f64(a->data, bcast_idx(a, out, i), a->dtype);
        double y = cml_load_f64(b->data, bcast_idx(b, out, i), b->dtype);
        double r;
        if (binary_f64(node->type, x, y, &r) != 0) return -1;
        cml_store_f64(out->data, i, out->dtype, r);
    }
    return 0;
}

/* --------------------------------------------------------- layout / movement
 *
 * These ops relocate elements without inspecting their values, so they copy
 * `esz` bytes at a time. That is exact for every dtype -- including int64 and
 * the fp8 formats, which the f64 domain could not represent losslessly. Index,
 * mask and condition operands are read through the f64 accessor, since those
 * are numeric regardless of how they are stored.
 */

#define ELEM_COPY(dst, di, src, si, esz) \
    memcpy((uint8_t*)(dst) + (di) * (esz), (const uint8_t*)(src) + (si) * (esz), (esz))

/* Broadcast source index for `flat_i`, mirroring _broadcast_idx in execution.c. */
static size_t bcast_idx(const Tensor* inp, const Tensor* out, size_t flat_i) {
    if (inp->numel == 1) return 0;
    if (inp->numel == out->numel) return flat_i;

    size_t out_str[16], inp_str[16];
    int ond = out->ndim, ind = inp->ndim;
    if (ond > 16 || ind > 16 || ond <= 0 || ind <= 0) return 0;
    out_str[ond - 1] = 1;
    for (int d = ond - 2; d >= 0; d--) out_str[d] = out_str[d + 1] * (size_t)out->shape[d + 1];
    inp_str[ind - 1] = 1;
    for (int d = ind - 2; d >= 0; d--) inp_str[d] = inp_str[d + 1] * (size_t)inp->shape[d + 1];

    size_t idx = 0, rem = flat_i;
    for (int d = 0; d < ond; d++) {
        size_t coord = rem / out_str[d];
        rem %= out_str[d];
        int id = d - (ond - ind);           /* right-align input dims */
        if (id >= 0 && inp->shape[id] > 1)
            idx += coord * inp_str[id];
    }
    return idx;
}

/* Read operand `i` element `j` as a number, whatever its dtype. */
static double num_at(struct IRNode* node, int i, size_t j) {
    Tensor* t = in_at(node, i);
    return (t && t->data) ? cml_load_f64(t->data, j, t->dtype) : 0.0;
}

/* Rank-agnostic index math for the layout kernels below. These mirror
 * pad_nd_f32/scatter_nd_f32/diagonal_nd_f32 in execution.c but move raw
 * elements, so one copy serves every dtype. Both paths were written with
 * ndim==1/2 special cases and no general arm; keep them general. */
static void strides_nd(const int* shape, int ndim, size_t* st) {
    size_t acc = 1;
    for (int d = ndim - 1; d >= 0; d--) { st[d] = acc; acc *= (size_t)shape[d]; }
}

static void pad_nd_typed(const void* ad, const Tensor* a, const Tensor* out, void* od,
                         const PadParams* p, size_t esz) {
    int nd = a->ndim;
    size_t in_str[8], out_str[8];
    strides_nd(a->shape, nd, in_str);
    strides_nd(out->shape, out->ndim, out_str);

    for (size_t i = 0; i < out->numel; i++) {
        size_t rem = i, src = 0;
        int ok = 1;
        for (int d = 0; d < nd; d++) {
            int coord = (int)(rem / out_str[d]);
            rem %= out_str[d];
            int sc = coord - p->pad_widths[2 * d];
            int n  = a->shape[d];
            if (sc < 0 || sc >= n) {
                if (p->mode == PAD_REFLECT) {
                    sc = (sc < 0) ? -sc : 2 * n - 2 - sc;
                    if (sc < 0) sc = 0;
                    if (sc >= n) sc = n - 1;
                } else if (p->mode == PAD_REPLICATE) {
                    sc = (sc < 0) ? 0 : n - 1;
                } else {
                    ok = 0;
                    break;
                }
            }
            src += (size_t)sc * in_str[d];
        }
        if (ok) ELEM_COPY(od, i, ad, src, esz);
        else    cml_store_f64(od, i, out->dtype, (double)p->value);
    }
}

static void scatter_nd_typed(struct IRNode* node, const void* ad, const Tensor* idx_t,
                             const void* src, const Tensor* out, void* od, int dim,
                             size_t esz) {
    int nd = out->ndim;
    memcpy(od, ad, out->numel * esz);
    if (nd <= 0 || nd > 8 || dim < 0 || dim >= nd) return;

    size_t out_str[8];
    strides_nd(out->shape, nd, out_str);

    for (size_t j = 0; j < idx_t->numel; j++) {
        size_t rem = j, off = 0;
        int ok = 1;
        for (int d = idx_t->ndim - 1; d >= 0; d--) {
            size_t c = rem % (size_t)idx_t->shape[d];
            rem /= (size_t)idx_t->shape[d];
            if (d >= nd) { ok = 0; break; }
            if (d == dim) c = (size_t)(int)num_at(node, 1, j);
            if (c >= (size_t)out->shape[d]) { ok = 0; break; }
            off += c * out_str[d];
        }
        if (ok && off < out->numel) ELEM_COPY(od, off, src, j, esz);
    }
}

static int diagonal_nd_typed(const void* ad, const Tensor* a, const Tensor* out, void* od,
                             int offset, int dim1, int dim2, size_t esz) {
    int nd = a->ndim;
    if (nd < 2 || nd > 8 || dim1 == dim2) return -1;
    if (dim1 < 0 || dim1 >= nd || dim2 < 0 || dim2 >= nd) return -1;

    size_t in_str[8];
    strides_nd(a->shape, nd, in_str);

    int map[8], diag_slot = 0, oi = 0;
    for (int d = 0; d < nd; d++) {
        if (d == dim2) continue;
        if (d == dim1) diag_slot = oi;
        map[oi++] = d;
    }

    for (size_t i = 0; i < out->numel; i++) {
        size_t rem = i, src = 0, k = 0;
        for (int j = oi - 1; j >= 0; j--) {
            size_t c = rem % (size_t)out->shape[j];
            rem /= (size_t)out->shape[j];
            if (j == diag_slot) k = c;
            else src += c * in_str[map[j]];
        }
        size_t r = (offset >= 0) ? k : k + (size_t)(-offset);
        size_t c = (offset >= 0) ? k + (size_t)offset : k;
        if (r >= (size_t)a->shape[dim1] || c >= (size_t)a->shape[dim2])
            cml_store_f64(od, i, out->dtype, 0.0);
        else
            ELEM_COPY(od, i, ad, src + r * in_str[dim1] + c * in_str[dim2], esz);
    }
    return 0;
}

static int layout_kernel(struct IRNode* node, Tensor* out, void* od) {
    Tensor* a     = in_at(node, 0);
    size_t  esz   = cml_dtype_size(out->dtype);
    if (!od || esz == 0) return -1;
    const void* ad = a ? a->data : NULL;
    size_t an = a ? a->numel : 0;

    switch (node->type) {

    /* --- creation: the lazy materializers already store through the output
     * dtype, so they need no typed variant, only routing. --- */
    case UOP_FILL: {
        FillParams* p = (FillParams*)node->params;
        return cml_cpu_lazy_fill(out, p ? p->value : 0.0f);
    }
    case UOP_CONST: {
        ConstParams* p = (ConstParams*)node->params;
        if (!p || !p->data) return -1;
        return cml_cpu_lazy_const(out, p->data, p->data_size);
    }
    case UOP_ARANGE_OP: {
        ArangeParams* p = (ArangeParams*)node->params;
        return p ? cml_cpu_lazy_arange(out, p->start, p->step) : -1;
    }
    case UOP_EYE_OP: {
        EyeParams* p = (EyeParams*)node->params;
        return p ? cml_cpu_lazy_eye(out, p->n) : -1;
    }
    case UOP_RAND_UNIFORM: return cml_cpu_lazy_rand_uniform(out);
    case UOP_RAND_NORMAL:  return cml_cpu_lazy_rand_normal(out);
    case UOP_RAND_INT: {
        RandIntParams* p = (RandIntParams*)node->params;
        return p ? cml_cpu_lazy_rand_int(out, p->low, p->high) : -1;
    }
    case UOP_ALLOC:
        return 0;  /* buffer already allocated; contents are scratch */

    /* --- straight copies --- */
    case UOP_RESHAPE:
    case UOP_FLATTEN:
    case UOP_UNFLATTEN:
        if (!ad) return -1;
        if (od == ad) return 0;                       /* view shares the buffer */
        if (an == out->numel) { memcpy(od, ad, out->numel * esz); return 0; }
        for (size_t i = 0; i < out->numel; i++) ELEM_COPY(od, i, ad, bidx(i, an), esz);
        return 0;

    case UOP_MESHGRID: {
        if (!ad || an == 0) return -1;
        for (size_t r = 0; r < out->numel / an; r++)
            memcpy((uint8_t*)od + r * an * esz, ad, an * esz);
        return 0;
    }

    case UOP_EXPAND: {
        if (!ad) return -1;
        if (an == out->numel) {
            if (od != ad) memcpy(od, ad, out->numel * esz);
            return 0;
        }
        /* Output may alias the input buffer while being larger than it, so it
         * needs storage of its own before the broadcast read/write. */
        if (od == ad) {
            void* nb = cml_buffer_cache_alloc(out->numel * esz);
            if (!nb) return -1;
            out->data = nb; out->owns_data = true; out->from_buffer_cache = true;
            od = nb;
        }
        for (size_t i = 0; i < out->numel; i++)
            ELEM_COPY(od, i, ad, bcast_idx(a, out, i), esz);
        return 0;
    }

    case UOP_PERMUTE: {
        PermuteParams* p = (PermuteParams*)node->params;
        int nd = a ? a->ndim : 0;
        if (!ad || !p || !p->perm || nd <= 0 || nd > 16) return -1;
        size_t in_str[16], pstr[16];
        size_t s = 1;
        for (int i = nd - 1; i >= 0; i--) { in_str[i] = s; s *= (size_t)a->shape[i]; }
        for (int i = 0; i < nd; i++) pstr[i] = in_str[p->perm[i]];
        int coord[16] = {0};
        size_t lin = 0;
        for (size_t o = 0; o < out->numel; o++) {
            ELEM_COPY(od, o, ad, lin, esz);
            for (int i = nd - 1; i >= 0; i--) {
                lin += pstr[i];
                if (++coord[i] < out->shape[i]) break;
                coord[i] = 0;
                lin -= pstr[i] * (size_t)out->shape[i];
            }
        }
        return 0;
    }

    case UOP_SLICE: {
        SliceParams* p = (SliceParams*)node->params;
        if (!ad) return -1;
        if (!p || !p->start || !p->end) {
            size_t n = out->numel < an ? out->numel : an;
            memcpy(od, ad, n * esz);
            return 0;
        }
        int nd = a->ndim;
        memset(od, 0, out->numel * esz);
        for (size_t i = 0; i < out->numel; i++) {
            size_t src = 0, rem = i;
            bool valid = true;
            for (int d = nd - 1; d >= 0; d--) {
                size_t coord = rem % (size_t)out->shape[d];
                rem /= (size_t)out->shape[d];
                int sc = p->start[d] + (int)coord * (p->step ? p->step[d] : 1);
                if (sc < 0 || sc >= a->shape[d]) { valid = false; break; }
                size_t str = 1;
                for (int dd = d + 1; dd < nd; dd++) str *= (size_t)a->shape[dd];
                src += (size_t)sc * str;
            }
            if (valid && src < an) ELEM_COPY(od, i, ad, src, esz);
        }
        return 0;
    }

    case UOP_SHRINK: {
        ShrinkParams* p = (ShrinkParams*)node->params;
        int nd = a ? a->ndim : 0;
        if (!ad || !p || nd <= 0 || nd > 16) return -1;
        size_t in_str[16], s = 1;
        for (int d = nd - 1; d >= 0; d--) { in_str[d] = s; s *= (size_t)a->shape[d]; }
        for (size_t i = 0; i < out->numel; i++) {
            size_t rem = i, src = 0;
            for (int d = nd - 1; d >= 0; d--) {
                int c = (int)(rem % (size_t)out->shape[d]);
                rem /= (size_t)out->shape[d];
                src += (size_t)(p->starts[d] + c) * in_str[d];
            }
            if (src < an) ELEM_COPY(od, i, ad, src, esz);
        }
        return 0;
    }

    case UOP_STRIDE: {
        StrideParams* p = (StrideParams*)node->params;
        if (!ad) return -1;
        if (!p || !p->new_strides || p->num_dims != out->ndim) {
            size_t n = out->numel < an ? out->numel : an;
            memcpy(od, ad, n * esz);
            return 0;
        }
        memset(od, 0, out->numel * esz);
        for (size_t i = 0; i < out->numel; i++) {
            size_t src = 0, rem = i;
            for (int d = out->ndim - 1; d >= 0; d--) {
                size_t coord = rem % (size_t)out->shape[d];
                rem /= (size_t)out->shape[d];
                src += coord * p->new_strides[d];
            }
            if (src < an) ELEM_COPY(od, i, ad, src, esz);
        }
        return 0;
    }

    case UOP_CAT: {
        /* Was ndim 1/2 only; rank 3+ fell through to -1, so every non-f32
         * concat on a higher-rank tensor failed the dispatcher. Each input
         * contributes a contiguous run of `shape[dim] * inner` elements per
         * outer slice, which is rank-agnostic. */
        CatParams* p = (CatParams*)node->params;
        int dim = axis_of(p ? p->dim : 0, out->ndim);
        if (dim < 0 || node->num_inputs < 1) return -1;

        size_t outer = 1, inner = 1;
        for (int d = 0; d < dim; d++) outer *= (size_t)out->shape[d];
        for (int d = dim + 1; d < out->ndim; d++) inner *= (size_t)out->shape[d];
        size_t out_count = (size_t)out->shape[dim];

        size_t placed = 0;
        for (int t = 0; t < node->num_inputs; t++) {
            Tensor* s = in_at(node, t);
            if (!s || !s->data || s->ndim != out->ndim) return -1;
            size_t cnt = (size_t)s->shape[dim];
            if (placed + cnt > out_count) return -1;
            for (size_t o = 0; o < outer; o++)
                memcpy((uint8_t*)od + (o * out_count * inner + placed * inner) * esz,
                       (const uint8_t*)s->data + (o * cnt * inner) * esz, cnt * inner * esz);
            placed += cnt;
        }
        return (placed == out_count) ? 0 : -1;
    }

    case UOP_SCATTER: {
        Tensor* idx = in_at(node, 1);
        Tensor* src = in_at(node, 2);
        if (!ad || !idx || !idx->data || !src || !src->data) return -1;
        {
            ScatterParams* p = (ScatterParams*)node->params;
            int dim = p ? p->dim : 0;
            if (dim < 0) dim += out->ndim;
            scatter_nd_typed(node, ad, idx, src->data, out, od, dim, esz);
        }
        return 0;
    }

    case UOP_GATHER: {
        /* out[o, j, k] = a[o, idx[j], k] — gather along `dim` with 1-D
         * indices of any dtype (read through cml_load_i64). Mirrors the
         * f32 kernel's layout in execution.c. */
        Tensor* idx = in_at(node, 1);
        if (!ad || !od || !idx || !idx->data || idx->numel == 0) return -1;
        GatherParams* p = (GatherParams*)node->params;
        int dim = p ? p->dim : -1;
        if (dim < 0) dim += a->ndim;
        if (dim < 0 || dim >= a->ndim) return -1;

        size_t outer = 1, inner = 1;
        for (int d = 0; d < dim; d++)             outer *= (size_t)a->shape[d];
        for (int d = dim + 1; d < a->ndim; d++)   inner *= (size_t)a->shape[d];
        size_t dim_size = (size_t)a->shape[dim];

        size_t oi = 0;
        for (size_t o = 0; o < outer; o++) {
            for (size_t j = 0; j < idx->numel; j++) {
                int64_t iv = cml_load_i64(idx->data, j, idx->dtype);
                if (iv < 0 || (size_t)iv >= dim_size) {
                    LOG_ERROR("typed GATHER: index %lld out of bounds [0, %zu)",
                              (long long)iv, dim_size);
                    return -1;
                }
                for (size_t k = 0; k < inner && oi < out->numel; k++, oi++)
                    ELEM_COPY(od, oi, ad,
                              o * dim_size * inner + (size_t)iv * inner + k, esz);
            }
        }
        return 0;
    }

    case UOP_ROLL: {
        RollParams* p = (RollParams*)node->params;
        if (!ad || !p) return -1;
        int dim = axis_of(a->ndim == 1 ? 0 : p->dim, a->ndim);
        if (dim < 0) return -1;
        size_t outer, cnt, inner;
        lanes_of(a, dim, &outer, &cnt, &inner);
        int s = (int)(((p->shift % (int)cnt) + (int)cnt) % (int)cnt);
        for (size_t o = 0; o < outer; o++)
            for (size_t j = 0; j < cnt; j++)
                for (size_t m = 0; m < inner; m++)
                    ELEM_COPY(od, (o * cnt + (j + (size_t)s) % cnt) * inner + m,
                              ad, (o * cnt + j) * inner + m, esz);
        return 0;
    }

    case UOP_TILE: {
        TileParams* p = (TileParams*)node->params;
        if (!ad || !p) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            size_t src = 0, rem = i, str = 1;
            for (int d = out->ndim - 1; d >= 0; d--) {
                int coord = (int)(rem % (size_t)out->shape[d]);
                rem /= (size_t)out->shape[d];
                src += (size_t)(coord % a->shape[d]) * str;
                str *= (size_t)a->shape[d];
            }
            ELEM_COPY(od, i, ad, src, esz);
        }
        return 0;
    }

    case UOP_REPEAT_INTERLEAVE: {
        RepeatInterleaveParams* p = (RepeatInterleaveParams*)node->params;
        if (!ad || !p || p->repeats <= 0) return -1;
        int dim = axis_of(a->ndim == 1 ? 0 : p->dim, a->ndim);
        if (dim < 0) return -1;
        size_t outer, cnt, inner;
        lanes_of(a, dim, &outer, &cnt, &inner);
        size_t reps = (size_t)p->repeats;
        for (size_t o = 0; o < outer; o++)
            for (size_t j = 0; j < cnt; j++)
                for (size_t r = 0; r < reps; r++)
                    for (size_t m = 0; m < inner; m++)
                        ELEM_COPY(od, ((o * cnt + j) * reps + r) * inner + m,
                                  ad, (o * cnt + j) * inner + m, esz);
        return 0;
    }

    case UOP_DIAG: {
        DiagParams* p = (DiagParams*)node->params;
        int off = p ? p->offset : 0;
        if (!ad) return -1;
        memset(od, 0, out->numel * esz);
        if (a->ndim == 1) {
            int n = out->shape[0];
            for (int i = 0; i < a->shape[0]; i++) {
                int r = (off >= 0) ? i : i - off, c = (off >= 0) ? i + off : i;
                if (r >= 0 && r < n && c >= 0 && c < n)
                    ELEM_COPY(od, (size_t)r * n + c, ad, (size_t)i, esz);
            }
            return 0;
        }
        if (a->ndim == 2) {
            int rows = a->shape[0], cols = a->shape[1];
            for (int i = 0, oi = 0; oi < (int)out->numel; i++) {
                int r = (off >= 0) ? i : i - off, c = (off >= 0) ? i + off : i;
                if (r < 0 || r >= rows || c < 0 || c >= cols) break;
                ELEM_COPY(od, (size_t)oi++, ad, (size_t)r * cols + c, esz);
            }
            return 0;
        }
        return -1;
    }

    case UOP_DIAGONAL: {
        DiagParams* p = (DiagParams*)node->params;
        if (!ad) return -1;
        return diagonal_nd_typed(ad, a, out, od, p ? p->offset : 0, p ? p->dim1 : 0,
                                 p ? p->dim2 : 1, esz);
    }

    case UOP_TRIU:
    case UOP_TRIL: {
        TriParams* p = (TriParams*)node->params;
        int diag = p ? p->diagonal : 0;
        if (!ad || a->ndim < 2) return -1;
        int rows = a->shape[a->ndim - 2], cols = a->shape[a->ndim - 1];
        size_t plane = (size_t)rows * (size_t)cols;
        bool upper = (node->type == UOP_TRIU);
        for (size_t b = 0; b < an / plane; b++)
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++) {
                    size_t i = b * plane + (size_t)r * cols + c;
                    bool keep = upper ? (c >= r + diag) : (c <= r + diag);
                    if (keep) ELEM_COPY(od, i, ad, i, esz);
                    else      cml_store_f64(od, i, out->dtype, 0.0);
                }
        return 0;
    }

    case UOP_WHERE: {
        Tensor* x = in_at(node, 1);
        Tensor* y = in_at(node, 2);
        Tensor* c = in_at(node, 0);
        if (!c || !x || !y || !c->data || !x->data || !y->data) return -1;
        for (size_t i = 0; i < out->numel; i++) {
            bool t = num_at(node, 0, bcast_idx(c, out, i)) != 0.0;
            Tensor* s = t ? x : y;
            size_t si = bcast_idx(s, out, i);
            if (s->dtype == out->dtype) ELEM_COPY(od, i, s->data, si, esz);
            else cml_store_f64(od, i, out->dtype, cml_load_f64(s->data, si, s->dtype));
        }
        return 0;
    }

    case UOP_MASKED_FILL: {
        Tensor* m = in_at(node, 1);
        MaskedFillParams* p = (MaskedFillParams*)node->params;
        if (!ad || !m || !m->data) return -1;
        double fv = p ? (double)p->value : 0.0;
        for (size_t i = 0; i < out->numel; i++) {
            if (num_at(node, 1, bcast_idx(m, out, i)) != 0.0)
                cml_store_f64(od, i, out->dtype, fv);
            else
                ELEM_COPY(od, i, ad, bcast_idx(a, out, i), esz);
        }
        return 0;
    }

    case UOP_MASKED_SELECT: {
        Tensor* m = in_at(node, 1);
        if (!ad || !m || !m->data) return -1;
        size_t cnt = 0;
        for (size_t i = 0; i < an && i < m->numel; i++)
            if (num_at(node, 1, i) != 0.0) ELEM_COPY(od, cnt++, ad, i, esz);
        /* Data-dependent length: tensor_data_ptr re-syncs the tensor's shape
         * from the node after execution, so shrinking only `out` is undone
         * and the caller sees stale elements past the real result. */
        for (size_t i = cnt; i < out->numel; i++)
            cml_store_f64(od, i, out->dtype, 0.0);
        if (node->output_shape && node->output_ndim == 1)
            node->output_shape[0] = (int)cnt;
        out->numel    = cnt;
        out->shape[0] = (int)cnt;
        return 0;
    }

    case UOP_ONE_HOT: {
        OneHotParams* p = (OneHotParams*)node->params;
        int nc = p ? p->num_classes : 0;
        if (!ad || nc <= 0) return -1;
        memset(od, 0, out->numel * esz);
        for (size_t i = 0; i < an; i++) {
            int cls = (int)num_at(node, 0, i);
            if (cls >= 0 && cls < nc)
                cml_store_f64(od, i * (size_t)nc + (size_t)cls, out->dtype, 1.0);
        }
        return 0;
    }

    case UOP_PAD: {
        PadParams* p = (PadParams*)node->params;
        if (!ad || !p || a->ndim <= 0 || a->ndim > 8) return -1;
        pad_nd_typed(ad, a, out, od, p, esz);
        return 0;
    }

    default:
        return -1;
    }
}

static int exec_layout(struct IRNode* node, Tensor* out) {
    return layout_kernel(node, out, out->data);
}

/* ------------------------------------------------- reductions and ordering
 *
 * All of these run in the f64 domain and share the (outer, count, inner)
 * decomposition, so a single body handles any rank and any axis -- the f32
 * kernels these mirror only implement the 2-D and global cases.
 */

/* Reduce axis: the params' first dim, or -1 for a whole-tensor reduction. */
static int reduce_dim_of(struct IRNode* node, const Tensor* a) {
    ReduceParams* rp = (ReduceParams*)node->params;
    if (!rp || !rp->dims || rp->num_dims <= 0) return -1;
    return axis_of(rp->dims[0], a->ndim);
}

static int exec_reduce_like(struct IRNode* node, Tensor* out) {
    Tensor* a = in_at(node, 0);
    if (!a || !a->data || !out->data) return -1;
    UOpType t = node->type;

    if (t == UOP_TRACE) {
        if (a->ndim != 2) return -1;
        int rows = a->shape[0], cols = a->shape[1];
        int n = rows < cols ? rows : cols;
        double s = 0;
        for (int i = 0; i < n; i++) s += cml_load_f64(a->data, (size_t)i * cols + i, a->dtype);
        cml_store_f64(out->data, 0, out->dtype, s);
        return 0;
    }

    switch (t) {
    case UOP_PROD: case UOP_LOGSUMEXP: case UOP_VAR: case UOP_STD:
    case UOP_ALL:  case UOP_ANY:       case UOP_ARGMAX: case UOP_ARGMIN:
        break;
    default:
        return -1;
    }

    int dim = reduce_dim_of(node, a);
    size_t outer = 1, cnt = a->numel, inner = 1;
    if (dim >= 0) lanes_of(a, dim, &outer, &cnt, &inner);
    if (cnt == 0) return -1;

    for (size_t o = 0; o < outer; o++) {
        for (size_t m = 0; m < inner; m++) {
            size_t base = o * cnt * inner + m;
            double acc;
            switch (t) {
            case UOP_PROD:
                acc = 1.0;
                for (size_t j = 0; j < cnt; j++)
                    acc *= cml_load_f64(a->data, base + j * inner, a->dtype);
                break;
            case UOP_ALL:
                acc = 1.0;
                for (size_t j = 0; j < cnt; j++)
                    if (cml_load_f64(a->data, base + j * inner, a->dtype) == 0.0) { acc = 0.0; break; }
                break;
            case UOP_ANY:
                acc = 0.0;
                for (size_t j = 0; j < cnt; j++)
                    if (cml_load_f64(a->data, base + j * inner, a->dtype) != 0.0) { acc = 1.0; break; }
                break;
            case UOP_LOGSUMEXP: {
                double mx = -INFINITY;
                for (size_t j = 0; j < cnt; j++) {
                    double v = cml_load_f64(a->data, base + j * inner, a->dtype);
                    if (v > mx) mx = v;
                }
                double s = 0;
                for (size_t j = 0; j < cnt; j++)
                    s += exp(cml_load_f64(a->data, base + j * inner, a->dtype) - mx);
                acc = isinf(mx) ? mx : mx + log(s);
                break;
            }
            case UOP_VAR: case UOP_STD: {
                double mean = 0;
                for (size_t j = 0; j < cnt; j++)
                    mean += cml_load_f64(a->data, base + j * inner, a->dtype);
                mean /= (double)cnt;
                double v = 0;
                for (size_t j = 0; j < cnt; j++) {
                    double d = cml_load_f64(a->data, base + j * inner, a->dtype) - mean;
                    v += d * d;
                }
                v /= (double)cnt;              /* biased, matching the f32 kernel */
                acc = (t == UOP_STD) ? sqrt(v) : v;
                break;
            }
            default: {                          /* ARGMAX / ARGMIN */
                bool want_max = (t == UOP_ARGMAX);
                double best = cml_load_f64(a->data, base, a->dtype);
                size_t bi = 0;
                for (size_t j = 1; j < cnt; j++) {
                    double v = cml_load_f64(a->data, base + j * inner, a->dtype);
                    if (want_max ? (v > best) : (v < best)) { best = v; bi = j; }
                }
                acc = (double)bi;
                break;
            }
            }
            size_t oi = o * inner + m;
            if (oi < out->numel) cml_store_f64(out->data, oi, out->dtype, acc);
        }
    }
    return 0;
}

static int exec_cumulative(struct IRNode* node, Tensor* out) {
    Tensor* a = in_at(node, 0);
    if (!a || !a->data || !out->data) return -1;
    UOpType t = node->type;
    switch (t) {
    case UOP_CUMSUM: case UOP_CUMPROD: case UOP_CUMMAX:
    case UOP_CUMMIN: case UOP_LOGCUMSUMEXP: break;
    default: return -1;
    }
    CumsumParams* p = (CumsumParams*)node->params;
    int dim = axis_of(p ? p->dim : (a->ndim - 1), a->ndim);
    if (dim < 0) return -1;

    size_t outer, cnt, inner;
    lanes_of(a, dim, &outer, &cnt, &inner);
    for (size_t o = 0; o < outer; o++)
        for (size_t m = 0; m < inner; m++) {
            size_t base = o * cnt * inner + m;
            double acc = (t == UOP_CUMPROD) ? 1.0
                       : (t == UOP_CUMMAX) ? -INFINITY
                       : (t == UOP_CUMMIN) ?  INFINITY
                       : (t == UOP_LOGCUMSUMEXP) ? -INFINITY : 0.0;
            for (size_t j = 0; j < cnt; j++) {
                size_t k = base + j * inner;
                double v = cml_load_f64(a->data, k, a->dtype);
                switch (t) {
                case UOP_CUMSUM:  acc += v; break;
                case UOP_CUMPROD: acc *= v; break;
                case UOP_CUMMAX:  if (v > acc) acc = v; break;
                case UOP_CUMMIN:  if (v < acc) acc = v; break;
                default: {        /* logcumsumexp, running max for stability */
                    double mx = fmax(acc, v);
                    acc = isinf(mx) && mx < 0 ? mx : mx + log(exp(acc - mx) + exp(v - mx));
                    break;
                }
                }
                cml_store_f64(out->data, k, out->dtype, acc);
            }
        }
    return 0;
}

/* Selection sort over one lane -- k passes for topk, full for sort. Matches the
 * f32 kernels' tie-breaking (first occurrence wins, since comparisons are
 * strict). */
static void lane_order(const void* src, DType dt, size_t base, size_t inner,
                       size_t cnt, size_t k, bool desc, size_t* ord) {
    for (size_t j = 0; j < cnt; j++) ord[j] = j;
    for (size_t i = 0; i < k && i < cnt; i++) {
        size_t best = i;
        double bv = cml_load_f64(src, base + ord[i] * inner, dt);
        for (size_t j = i + 1; j < cnt; j++) {
            double v = cml_load_f64(src, base + ord[j] * inner, dt);
            if (desc ? (v > bv) : (v < bv)) { best = j; bv = v; }
        }
        size_t sw = ord[i]; ord[i] = ord[best]; ord[best] = sw;
    }
}

static int exec_order(struct IRNode* node, Tensor* out) {
    Tensor* a = in_at(node, 0);
    if (!a || !a->data || !out->data) return -1;

    int dim; size_t k; bool desc; bool want_idx;
    if (node->type == UOP_SORT || node->type == UOP_ARGSORT) {
        SortParams* p = (SortParams*)node->params;
        dim  = axis_of(p ? p->dim : (a->ndim - 1), a->ndim);
        desc = p ? p->descending : false;
        want_idx = (node->type == UOP_ARGSORT);
        k = (dim >= 0) ? (size_t)a->shape[dim] : 0;
    } else if (node->type == UOP_TOPK) {
        TopkParams* p = (TopkParams*)node->params;
        dim  = axis_of(p ? p->dim : (a->ndim - 1), a->ndim);
        desc = p ? p->largest : true;
        want_idx = false;
        k = p ? (size_t)p->k : 1;
    } else {
        return -1;
    }
    if (dim < 0) return -1;

    size_t outer, cnt, inner;
    lanes_of(a, dim, &outer, &cnt, &inner);
    if (k > cnt) k = cnt;
    size_t* ord = (size_t*)cml_malloc(cnt * sizeof(size_t));
    if (!ord) return -1;

    for (size_t o = 0; o < outer; o++)
        for (size_t m = 0; m < inner; m++) {
            size_t base = o * cnt * inner + m;
            lane_order(a->data, a->dtype, base, inner, cnt, k, desc, ord);
            for (size_t i = 0; i < k; i++) {
                size_t oi = (o * k + i) * inner + m;
                if (oi >= out->numel) continue;
                if (want_idx)
                    cml_store_f64(out->data, oi, out->dtype, (double)ord[i]);
                else
                    cml_store_f64(out->data, oi, out->dtype,
                                  cml_load_f64(a->data, base + ord[i] * inner, a->dtype));
            }
        }
    cml_free(ord);
    return 0;
}

/* ------------------------------------------------------- remaining compute */

static int exec_compute(struct IRNode* node, Tensor* out) {
    Tensor* a = in_at(node, 0);
    void* od  = out->data;
    if (!od) return -1;

    switch (node->type) {

    case UOP_NONZERO: {
        if (!a || !a->data) return -1;
        int idx = 0;
        if (a->ndim == 1) {
            for (size_t i = 0; i < a->numel; i++)
                if (num_at(node, 0, i) != 0.0 && idx < (int)out->numel)
                    cml_store_f64(od, (size_t)idx++, out->dtype, (double)i);
        } else if (a->ndim == 2) {
            int cols = a->shape[1];
            for (size_t i = 0; i < a->numel; i++)
                if (num_at(node, 0, i) != 0.0 && idx + 1 < (int)out->numel) {
                    cml_store_f64(od, (size_t)idx++, out->dtype, (double)((int)i / cols));
                    cml_store_f64(od, (size_t)idx++, out->dtype, (double)((int)i % cols));
                }
        } else {
            return -1;
        }
        for (int i = idx; i < (int)out->numel; i++)
            cml_store_f64(od, (size_t)i, out->dtype, -1.0);
        return 0;
    }

    case UOP_LERP: {
        Tensor* b = in_at(node, 1);
        Tensor* tt = in_at(node, 2);
        if (!a || !b || !a->data || !b->data) return -1;
        bool tensor_t = tt && tt->data;
        for (size_t i = 0; i < out->numel; i++) {
            double x = num_at(node, 0, bcast_idx(a, out, i));
            double y = num_at(node, 1, bcast_idx(b, out, i));
            double w = tensor_t ? num_at(node, 2, bcast_idx(tt, out, i)) : 0.5;
            cml_store_f64(od, i, out->dtype, x + w * (y - x));
        }
        return 0;
    }

    case UOP_LINEAR: {
        Tensor* w = in_at(node, 1);
        Tensor* b = in_at(node, 2);
        if (!a || !w || !a->data || !w->data) return -1;
        if (a->ndim < 2 || w->ndim != 2) return -1;
        size_t K = (size_t)a->shape[a->ndim - 1];
        size_t N = (size_t)w->shape[0];
        size_t rows = a->numel / K;
        for (size_t m = 0; m < rows; m++)
            for (size_t n = 0; n < N; n++) {
                double s = 0;
                for (size_t k = 0; k < K; k++)
                    s += cml_load_f64(a->data, m * K + k, a->dtype) *
                         cml_load_f64(w->data, n * K + k, w->dtype);
                if (b && b->data) s += cml_load_f64(b->data, n, b->dtype);
                cml_store_f64(od, m * N + n, out->dtype, s);
            }
        return 0;
    }

    case UOP_MAXPOOL2D:
    case UOP_AVGPOOL2D: {
        Pool2DParams* p = (Pool2DParams*)node->params;
        if (!a || !a->data || !p || a->ndim != 4) return -1;
        bool is_max = (node->type == UOP_MAXPOOL2D);
        int N = a->shape[0], C = a->shape[1], H = a->shape[2], W = a->shape[3];
        int kh = p->kernel_size[0], kw = p->kernel_size[1];
        int sh = p->stride[0] > 0 ? p->stride[0] : kh;
        int sw = p->stride[1] > 0 ? p->stride[1] : kw;
        int ph = p->padding[0], pw = p->padding[1];
        int dh = p->dilation[0] > 0 ? p->dilation[0] : 1;
        int dw = p->dilation[1] > 0 ? p->dilation[1] : 1;
        int OH = out->shape[2], OW = out->shape[3];
        for (int n = 0; n < N; n++)
          for (int c = 0; c < C; c++)
            for (int oh = 0; oh < OH; oh++)
              for (int ow = 0; ow < OW; ow++) {
                double acc = is_max ? -INFINITY : 0.0;
                int count = 0;
                for (int i = 0; i < kh; i++) {
                    int ih = oh * sh - ph + i * dh;
                    for (int j = 0; j < kw; j++) {
                        int iw = ow * sw - pw + j * dw;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                            if (!is_max && p->count_include_pad) count++;
                            continue;
                        }
                        size_t si = (((size_t)n * C + c) * H + ih) * W + iw;
                        double v = cml_load_f64(a->data, si, a->dtype);
                        if (is_max) { if (v > acc) acc = v; }
                        else        { acc += v; count++; }
                    }
                }
                size_t oi = (((size_t)n * C + c) * OH + oh) * OW + ow;
                cml_store_f64(od, oi, out->dtype,
                              is_max ? acc : (count ? acc / count : 0.0));
              }
        return 0;
    }

    case UOP_UNFOLD: {
        UnfoldParams* p = (UnfoldParams*)node->params;
        if (!a || !a->data || !p || p->stride <= 0) return -1;
        int ks = p->kernel_size, st = p->stride;
        int last = a->shape[a->ndim - 1];
        int nw = (last - ks) / st + 1;
        size_t batch = 1;
        for (int d = 0; d < a->ndim - 1; d++) batch *= (size_t)a->shape[d];
        for (size_t b = 0; b < batch; b++)
            for (int w = 0; w < nw; w++)
                for (int k = 0; k < ks; k++)
                    cml_store_f64(od, (b * (size_t)nw + w) * ks + k, out->dtype,
                                  cml_load_f64(a->data, b * (size_t)last + w * st + k, a->dtype));
        return 0;
    }

    case UOP_FOLD: {
        FoldParams* p = (FoldParams*)node->params;
        if (!a || !a->data || !p) return -1;
        int ks = p->kernel_size, st = p->stride, L = p->output_len;
        int nd = a->ndim, nw = a->shape[nd - 2];
        size_t batch = 1;
        for (int d = 0; d < nd - 2; d++) batch *= (size_t)a->shape[d];
        for (size_t i = 0; i < out->numel; i++) cml_store_f64(od, i, out->dtype, 0.0);
        for (size_t b = 0; b < batch; b++)
            for (int w = 0; w < nw; w++)
                for (int k = 0; k < ks; k++) {
                    size_t oi = b * (size_t)L + (size_t)(w * st + k);
                    cml_store_f64(od, oi, out->dtype,
                                  cml_load_f64(od, oi, out->dtype) +
                                  cml_load_f64(a->data, (b * (size_t)nw + w) * ks + k, a->dtype));
                }
        return 0;
    }

    case UOP_IM2COL: {
        Im2colParams* p = (Im2colParams*)node->params;
        if (!a || !a->data || !p || a->ndim != 4) return -1;
        int N = a->shape[0], C = a->shape[1], H = a->shape[2], W = a->shape[3];
        int OH = (H + 2 * p->ph - p->dh * (p->kh - 1) - 1) / p->sh + 1;
        int OW = (W + 2 * p->pw - p->dw * (p->kw - 1) - 1) / p->sw + 1;
        size_t K = (size_t)C * p->kh * p->kw;
        for (size_t i = 0; i < out->numel; i++) cml_store_f64(od, i, out->dtype, 0.0);
        for (int n = 0; n < N; n++)
          for (int c = 0; c < C; c++)
            for (int ki = 0; ki < p->kh; ki++)
              for (int kj = 0; kj < p->kw; kj++) {
                size_t col = ((size_t)c * p->kh + ki) * p->kw + kj;
                for (int oh = 0; oh < OH; oh++) {
                    int ih = oh * p->sh + ki * p->dh - p->ph;
                    if (ih < 0 || ih >= H) continue;
                    for (int ow = 0; ow < OW; ow++) {
                        int iw = ow * p->sw + kj * p->dw - p->pw;
                        if (iw < 0 || iw >= W) continue;
                        size_t row = (size_t)(n * OH + oh) * OW + ow;
                        size_t si  = (((size_t)n * C + c) * H + ih) * W + iw;
                        cml_store_f64(od, row * K + col, out->dtype,
                                      cml_load_f64(a->data, si, a->dtype));
                    }
                }
              }
        return 0;
    }

    case UOP_COL2IM: {
        Col2imParams* p = (Col2imParams*)node->params;
        if (!a || !a->data || !p) return -1;
        int C = p->C, H = p->H, W = p->W;
        int OH = (H + 2 * p->ph - p->dh * (p->kh - 1) - 1) / p->sh + 1;
        int OW = (W + 2 * p->pw - p->dw * (p->kw - 1) - 1) / p->sw + 1;
        size_t K = (size_t)C * p->kh * p->kw;
        int N = (OH * OW > 0) ? (int)(a->shape[0] / (OH * OW)) : 0;
        for (size_t i = 0; i < out->numel; i++) cml_store_f64(od, i, out->dtype, 0.0);
        for (int n = 0; n < N; n++)
          for (int c = 0; c < C; c++)
            for (int ki = 0; ki < p->kh; ki++)
              for (int kj = 0; kj < p->kw; kj++) {
                size_t col = ((size_t)c * p->kh + ki) * p->kw + kj;
                for (int oh = 0; oh < OH; oh++) {
                    int ih = oh * p->sh + ki * p->dh - p->ph;
                    if (ih < 0 || ih >= H) continue;
                    for (int ow = 0; ow < OW; ow++) {
                        int iw = ow * p->sw + kj * p->dw - p->pw;
                        if (iw < 0 || iw >= W) continue;
                        size_t row = (size_t)(n * OH + oh) * OW + ow;
                        size_t oi  = (((size_t)n * C + c) * H + ih) * W + iw;
                        cml_store_f64(od, oi, out->dtype,
                                      cml_load_f64(od, oi, out->dtype) +
                                      cml_load_f64(a->data, row * K + col, a->dtype));
                    }
                }
              }
        return 0;
    }

    case UOP_SCATTER_ADD: {
        ScatterAddParams* p = (ScatterAddParams*)node->params;
        Tensor* src = in_at(node, 1);
        if (!a || !a->data || !src || !src->data || !p) return -1;
        int dim = p->dim;
        size_t outer = 1, inner = 1;
        for (int d = 0; d < dim; d++) outer *= (size_t)src->shape[d];
        for (int d = dim + 1; d < src->ndim; d++) inner *= (size_t)src->shape[d];
        size_t sdim = (size_t)src->shape[dim], odim = (size_t)p->dim_size;
        for (size_t i = 0; i < out->numel; i++) cml_store_f64(od, i, out->dtype, 0.0);
        for (size_t o = 0; o < outer; o++)
            for (size_t j = 0; j < sdim; j++) {
                int k = (int)num_at(node, 0, dim == 0 ? j : (o * sdim + j));
                if (k < 0 || k >= (int)odim) continue;
                for (size_t m = 0; m < inner; m++) {
                    size_t oi = (o * odim + (size_t)k) * inner + m;
                    cml_store_f64(od, oi, out->dtype,
                                  cml_load_f64(od, oi, out->dtype) +
                                  cml_load_f64(src->data, (o * sdim + j) * inner + m, src->dtype));
                }
            }
        return 0;
    }

    case UOP_CONV3D: {
        Conv3DParams* p = (Conv3DParams*)node->params;
        Tensor* w = in_at(node, 1);
        Tensor* bi = in_at(node, 2);
        if (!a || !w || !a->data || !w->data || !p) return -1;
        int B = a->shape[0], IC = a->shape[1], ID = a->shape[2], IH = a->shape[3], IW = a->shape[4];
        int OC = w->shape[0], KD = w->shape[2], KH = w->shape[3], KW = w->shape[4];
        int OD = out->shape[2], OH = out->shape[3], OW = out->shape[4];
        for (int b = 0; b < B; b++)
          for (int oc = 0; oc < OC; oc++)
            for (int odp = 0; odp < OD; odp++)
              for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++) {
                  double s = (bi && bi->data) ? cml_load_f64(bi->data, (size_t)oc, bi->dtype) : 0.0;
                  for (int ic = 0; ic < IC; ic++)
                    for (int kd = 0; kd < KD; kd++) {
                      int iz = odp * p->stride[0] - p->padding[0] + kd * p->dilation[0];
                      if (iz < 0 || iz >= ID) continue;
                      for (int kh = 0; kh < KH; kh++) {
                        int iy = oh * p->stride[1] - p->padding[1] + kh * p->dilation[1];
                        if (iy < 0 || iy >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                          int ix = ow * p->stride[2] - p->padding[2] + kw * p->dilation[2];
                          if (ix < 0 || ix >= IW) continue;
                          size_t ii = ((((size_t)b * IC + ic) * ID + iz) * IH + iy) * IW + ix;
                          size_t wi = ((((size_t)oc * IC + ic) * KD + kd) * KH + kh) * KW + kw;
                          s += cml_load_f64(a->data, ii, a->dtype) *
                               cml_load_f64(w->data, wi, w->dtype);
                        }
                      }
                    }
                  size_t oi = ((((size_t)b * OC + oc) * OD + odp) * OH + oh) * OW + ow;
                  cml_store_f64(od, oi, out->dtype, s);
                }
        return 0;
    }

    case UOP_CONV_TRANSPOSE2D: {
        ConvTranspose2DParams* p = (ConvTranspose2DParams*)node->params;
        Tensor* w = in_at(node, 1);
        Tensor* bi = in_at(node, 2);
        if (!a || !w || !a->data || !w->data || !p) return -1;
        int B = a->shape[0], IC = a->shape[1], IH = a->shape[2], IW = a->shape[3];
        int OC = w->shape[1], KH = w->shape[2], KW = w->shape[3];
        int OH = out->shape[2], OW = out->shape[3];
        for (size_t i = 0; i < out->numel; i++) cml_store_f64(od, i, out->dtype, 0.0);
        for (int b = 0; b < B; b++)
          for (int ic = 0; ic < IC; ic++)
            for (int ih = 0; ih < IH; ih++)
              for (int iw = 0; iw < IW; iw++) {
                double v = cml_load_f64(a->data,
                             (((size_t)b * IC + ic) * IH + ih) * IW + iw, a->dtype);
                for (int oc = 0; oc < OC; oc++)
                  for (int kh = 0; kh < KH; kh++) {
                    int oh = ih * p->stride[0] - p->padding[0] + kh * p->dilation[0];
                    if (oh < 0 || oh >= OH) continue;
                    for (int kw = 0; kw < KW; kw++) {
                      int ow = iw * p->stride[1] - p->padding[1] + kw * p->dilation[1];
                      if (ow < 0 || ow >= OW) continue;
                      size_t wi = (((size_t)ic * OC + oc) * KH + kh) * KW + kw;
                      size_t oi = (((size_t)b * OC + oc) * OH + oh) * OW + ow;
                      cml_store_f64(od, oi, out->dtype,
                                    cml_load_f64(od, oi, out->dtype) +
                                    v * cml_load_f64(w->data, wi, w->dtype));
                    }
                  }
              }
        if (bi && bi->data)
            for (int b = 0; b < B; b++)
              for (int oc = 0; oc < OC; oc++)
                for (int oh = 0; oh < OH; oh++)
                  for (int ow = 0; ow < OW; ow++) {
                    size_t oi = (((size_t)b * OC + oc) * OH + oh) * OW + ow;
                    cml_store_f64(od, oi, out->dtype,
                                  cml_load_f64(od, oi, out->dtype) +
                                  cml_load_f64(bi->data, (size_t)oc, bi->dtype));
                  }
        return 0;
    }

    case UOP_CONV_TRANSPOSE3D: {
        ConvTranspose3DParams* p = (ConvTranspose3DParams*)node->params;
        Tensor* w = in_at(node, 1);
        Tensor* bi = in_at(node, 2);
        if (!a || !w || !a->data || !w->data || !p) return -1;
        int B = a->shape[0], IC = a->shape[1], ID = a->shape[2], IH = a->shape[3], IW = a->shape[4];
        int OC = w->shape[1], KD = w->shape[2], KH = w->shape[3], KW = w->shape[4];
        int OD = out->shape[2], OH = out->shape[3], OW = out->shape[4];
        for (size_t i = 0; i < out->numel; i++) cml_store_f64(od, i, out->dtype, 0.0);
        for (int b = 0; b < B; b++)
          for (int ic = 0; ic < IC; ic++)
            for (int id = 0; id < ID; id++)
              for (int ih = 0; ih < IH; ih++)
                for (int iw = 0; iw < IW; iw++) {
                  double v = cml_load_f64(a->data,
                      ((((size_t)b * IC + ic) * ID + id) * IH + ih) * IW + iw, a->dtype);
                  for (int oc = 0; oc < OC; oc++)
                    for (int kd = 0; kd < KD; kd++) {
                      int odp = id * p->stride[0] - p->padding[0] + kd * p->dilation[0];
                      if (odp < 0 || odp >= OD) continue;
                      for (int kh = 0; kh < KH; kh++) {
                        int oh = ih * p->stride[1] - p->padding[1] + kh * p->dilation[1];
                        if (oh < 0 || oh >= OH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                          int ow = iw * p->stride[2] - p->padding[2] + kw * p->dilation[2];
                          if (ow < 0 || ow >= OW) continue;
                          size_t wi = ((((size_t)ic * OC + oc) * KD + kd) * KH + kh) * KW + kw;
                          size_t oi = ((((size_t)b * OC + oc) * OD + odp) * OH + oh) * OW + ow;
                          cml_store_f64(od, oi, out->dtype,
                                        cml_load_f64(od, oi, out->dtype) +
                                        v * cml_load_f64(w->data, wi, w->dtype));
                        }
                      }
                    }
                }
        if (bi && bi->data)
            for (size_t i = 0; i < out->numel; i++) {
                size_t oc = (i / ((size_t)OD * OH * OW)) % (size_t)OC;
                cml_store_f64(od, i, out->dtype,
                              cml_load_f64(od, i, out->dtype) +
                              cml_load_f64(bi->data, oc, bi->dtype));
            }
        return 0;
    }

    case UOP_SGD_STEP: {
        SgdStepParams* p = (SgdStepParams*)node->params;
        Tensor* g = in_at(node, 1);
        Tensor* buf = in_at(node, 2);
        if (!p || !a || !a->data || !g || !g->data) return -1;
        for (size_t j = 0; j < out->numel; j++) {
            double pv = cml_load_f64(a->data, j, a->dtype);
            double gv = cml_load_f64(g->data, j, g->dtype) + (double)p->weight_decay * pv;
            double upd;
            if (p->momentum > 0.0f && buf && buf->data) {
                double b = (double)p->momentum * cml_load_f64(buf->data, j, buf->dtype) +
                           (1.0 - (double)p->dampening) * gv;
                cml_store_f64(buf->data, j, buf->dtype, b);
                upd = p->nesterov ? (gv + (double)p->momentum * b) : b;
            } else {
                upd = gv;
            }
            cml_store_f64(od, j, out->dtype, pv - (double)p->lr * upd);
        }
        return 0;
    }

    case UOP_ADAM_STEP: {
        AdamStepParams* p = (AdamStepParams*)node->params;
        Tensor* g   = in_at(node, 1);
        Tensor* m1  = in_at(node, 2);
        Tensor* m2  = in_at(node, 3);
        Tensor* mx  = in_at(node, 4);
        if (!p || !a || !a->data || !g || !g->data || !m1 || !m1->data || !m2 || !m2->data)
            return -1;
        double b1 = p->beta1, b2 = p->beta2;
        double bc1 = 1.0 - pow(b1, (double)p->step);
        double bc2 = 1.0 - pow(b2, (double)p->step);
        double lr_t = (double)p->lr * sqrt(bc2) / bc1;
        for (size_t j = 0; j < out->numel; j++) {
            double pv = cml_load_f64(a->data, j, a->dtype);
            double gv = cml_load_f64(g->data, j, g->dtype) + (double)p->weight_decay * pv;
            double e1 = b1 * cml_load_f64(m1->data, j, m1->dtype) + (1.0 - b1) * gv;
            double e2 = b2 * cml_load_f64(m2->data, j, m2->dtype) + (1.0 - b2) * gv * gv;
            cml_store_f64(m1->data, j, m1->dtype, e1);
            cml_store_f64(m2->data, j, m2->dtype, e2);
            double denom;
            if (mx && mx->data) {
                double mv = fmax(cml_load_f64(mx->data, j, mx->dtype), e2);
                cml_store_f64(mx->data, j, mx->dtype, mv);
                denom = sqrt(mv) + (double)p->eps;
            } else {
                denom = sqrt(e2) + (double)p->eps;
            }
            cml_store_f64(od, j, out->dtype, pv - lr_t * e1 / denom);
        }
        return 0;
    }

    case UOP_FUSED_ELEMENTWISE: {
        /* Same straight-line chain as the f32 kernel, evaluated one element at a
         * time in double. Operand refs: >=0 external input, <0 prior step. */
        FusedElementwiseParams* p = (FusedElementwiseParams*)node->params;
        if (!p || p->num_steps <= 0 || p->num_steps > 256) return -1;
        int ns = p->num_steps;
        double* step = (double*)cml_malloc((size_t)ns * sizeof(double));
        if (!step) return -1;

        for (size_t i = 0; i < out->numel; i++) {
            for (int s = 0; s < ns; s++) {
                double va = 0, vb = 0, vc = 0;
                int refs[3] = {p->a[s], p->b[s], p->c[s]};
                double* dst[3] = {&va, &vb, &vc};
                for (int r = 0; r < 3; r++) {
                    int ref = refs[r];
                    if (ref == FUSED_UNUSED_REF) continue;
                    if (ref < 0) { *dst[r] = step[-ref - 1]; continue; }
                    Tensor* it = in_at(node, ref);
                    if (!it || !it->data || it->numel == 0) continue;
                    *dst[r] = cml_load_f64(it->data, bidx(i, it->numel), it->dtype);
                }
                double r;
                UOpType t = p->op[s];
                if (t == UOP_FILL)       r = (double)p->konst[s];
                else if (t == UOP_WHERE) r = (va != 0.0) ? vb : vc;
                else if (binary_f64(t, va, vb, &r) != 0 &&
                         unary_f64(t, va, &r, node) != 0) {
                    cml_free(step);
                    return -1;
                }
                step[s] = r;
            }
            cml_store_f64(od, i, out->dtype, step[ns - 1]);
        }
        cml_free(step);
        return 0;
    }

    default:
        return -1;
    }
}

/* ------------------------------------------------------------ entry point */

static int dispatch(struct IRNode* node, Tensor* out) {
    if (exec_layout(node, out) == 0)      return 0;
    if (exec_cumulative(node, out) == 0)  return 0;
    if (exec_reduce_like(node, out) == 0) return 0;
    if (exec_order(node, out) == 0)       return 0;
    if (exec_compute(node, out) == 0)     return 0;
    if (node->num_inputs == 1 && exec_unary(node, out) == 0)  return 0;
    if (node->num_inputs == 2 && exec_binary(node, out) == 0) return 0;
    return -1;
}

/* An output tensor is frequently a view sharing an input's buffer. Writing such
 * an output in place corrupts the source mid-kernel: a zero-fill of the
 * destination is a zero-fill of the input, and a sort writes lane elements it
 * has not read yet. Aliasing nodes are therefore computed into scratch and
 * copied back -- an extra copy on a path that is already the slow one. */
static bool writes_may_clobber_reads(struct IRNode* node) {
    switch (node->type) {
    case UOP_RESHAPE: case UOP_FLATTEN: case UOP_UNFLATTEN:
        return false;   /* identity when the buffer is shared */
    case UOP_EXPAND:
        return false;   /* allocates its own output before writing */
    default:
        return true;
    }
}

int cml_exec_typed(struct IRNode* node, Tensor* out) {
    if (!node || !out || !out->data) return -1;

    bool aliases = false;
    for (int i = 0; i < node->num_inputs && node->inputs; i++)
        if (node->inputs[i] && node->inputs[i]->data == out->data) aliases = true;

    if (!aliases || !writes_may_clobber_reads(node))
        return dispatch(node, out);

    size_t esz = cml_dtype_size(out->dtype);
    void*  saved = out->data;
    void*  tmp   = cml_malloc(out->numel * esz);
    if (!tmp) return -1;
    out->data = tmp;
    int rc = dispatch(node, out);
    /* A kernel may have swapped in a buffer of its own (EXPAND); copy back from
     * wherever the result actually landed. */
    void* produced = out->data;
    out->data = saved;
    if (rc == 0 && produced) memcpy(saved, produced, out->numel * esz);
    if (produced == tmp) cml_free(tmp);
    return rc;
}
