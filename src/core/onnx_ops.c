#include "core/onnx.h"
#include "core/logging.h"
#include "ops/uops.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "alloc/cml_allocator.h"

#define TENSOR_MAP_SIZE 2048

typedef struct {
    char   name[128];
    Tensor *tensor;
    bool   occupied;
} TensorMapEntry;

typedef struct {
    TensorMapEntry entries[TENSOR_MAP_SIZE];
} TensorMap;

static uint32_t tensor_map_hash(const char *name)
{
    uint32_t h = 2166136261u;
    for (const char *p = name; *p; p++) {
        h ^= (uint32_t)(uint8_t)*p;
        h *= 16777619u;
    }
    return h;
}

static Tensor *tensor_map_get(TensorMap *map, const char *name)
{
    if (!name || !name[0]) return NULL;
    uint32_t idx = tensor_map_hash(name) % TENSOR_MAP_SIZE;
    for (int probe = 0; probe < TENSOR_MAP_SIZE; probe++) {
        uint32_t slot = (idx + (uint32_t)probe) % TENSOR_MAP_SIZE;
        if (!map->entries[slot].occupied) return NULL;
        if (strcmp(map->entries[slot].name, name) == 0) {
            return map->entries[slot].tensor;
        }
    }
    return NULL;
}

static void tensor_map_set(TensorMap *map, const char *name, Tensor *t)
{
    if (!name || !name[0]) return;
    uint32_t idx = tensor_map_hash(name) % TENSOR_MAP_SIZE;
    for (int probe = 0; probe < TENSOR_MAP_SIZE; probe++) {
        uint32_t slot = (idx + (uint32_t)probe) % TENSOR_MAP_SIZE;
        if (!map->entries[slot].occupied ||
            strcmp(map->entries[slot].name, name) == 0) {
            size_t len = strlen(name);
            if (len >= sizeof(map->entries[slot].name))
                len = sizeof(map->entries[slot].name) - 1;
            memcpy(map->entries[slot].name, name, len);
            map->entries[slot].name[len] = '\0';
            map->entries[slot].tensor   = t;
            map->entries[slot].occupied = true;
            return;
        }
    }
    LOG_ERROR("onnx_ops: tensor map full");
}

static const CMLONNXAttribute *find_attr(const CMLONNXNode *node,
                                          const char *name)
{
    for (int i = 0; i < node->num_attrs; i++) {
        if (strcmp(node->attrs[i].name, name) == 0) {
            return &node->attrs[i];
        }
    }
    return NULL;
}

static int64_t attr_int(const CMLONNXNode *node, const char *name,
                         int64_t def)
{
    const CMLONNXAttribute *a = find_attr(node, name);
    if (a && a->type == CML_ONNX_ATTR_INT) return a->value.i;
    return def;
}

static float attr_float(const CMLONNXNode *node, const char *name,
                          float def)
{
    const CMLONNXAttribute *a = find_attr(node, name);
    if (a && a->type == CML_ONNX_ATTR_FLOAT) return a->value.f;
    return def;
}

static const int64_t *attr_ints(const CMLONNXNode *node, const char *name,
                                 int *count)
{
    const CMLONNXAttribute *a = find_attr(node, name);
    if (a && a->type == CML_ONNX_ATTR_INTS) {
        if (count) *count = a->value.ints.count;
        return a->value.ints.data;
    }
    if (count) *count = 0;
    return NULL;
}

static bool attr_string(const CMLONNXNode *node, const char *name,
                        char *dst, size_t dst_size)
{
    const CMLONNXAttribute *a = find_attr(node, name);
    if (!a || a->type != CML_ONNX_ATTR_STRING) return false;
    size_t len = a->value.s.len;
    if (len >= dst_size) len = dst_size - 1;
    memcpy(dst, a->value.s.data, len);
    dst[len] = '\0';
    return true;
}

typedef Tensor *(*onnx_op_fn)(const CMLONNXNode *node, TensorMap *map);

static Tensor *inp(const CMLONNXNode *node, TensorMap *map, int idx)
{
    if (idx < 0 || idx >= node->num_inputs) return NULL;
    if (!node->inputs[idx] || node->inputs[idx][0] == '\0') return NULL;
    return tensor_map_get(map, node->inputs[idx]);
}

/* Realized float payload of an optional input; NULL when absent. */
static const float *tensor_floats(Tensor *t, int *count)
{
    if (!t) return NULL;
    tensor_ensure_executed(t);
    if (count) *count = (int)t->numel;
    return (const float *)tensor_data_ptr(t);
}


static Tensor *op_add(const CMLONNXNode *n, TensorMap *m)
{ return uop_add(inp(n,m,0), inp(n,m,1)); }

static Tensor *op_sub(const CMLONNXNode *n, TensorMap *m)
{ return uop_sub(inp(n,m,0), inp(n,m,1)); }

static Tensor *op_mul(const CMLONNXNode *n, TensorMap *m)
{ return uop_mul(inp(n,m,0), inp(n,m,1)); }

static Tensor *op_div(const CMLONNXNode *n, TensorMap *m)
{ return uop_div(inp(n,m,0), inp(n,m,1)); }


static Tensor *op_matmul(const CMLONNXNode *n, TensorMap *m)
{ return uop_matmul(inp(n,m,0), inp(n,m,1)); }


static Tensor *op_relu(const CMLONNXNode *n, TensorMap *m)
{ return uop_relu(inp(n,m,0)); }

static Tensor *op_sigmoid(const CMLONNXNode *n, TensorMap *m)
{ return uop_sigmoid(inp(n,m,0)); }

static Tensor *op_tanh(const CMLONNXNode *n, TensorMap *m)
{ return uop_tanh(inp(n,m,0)); }


static Tensor *op_exp(const CMLONNXNode *n, TensorMap *m)
{ return uop_exp(inp(n,m,0)); }

static Tensor *op_log(const CMLONNXNode *n, TensorMap *m)
{ return uop_log(inp(n,m,0)); }

static Tensor *op_sqrt(const CMLONNXNode *n, TensorMap *m)
{ return uop_sqrt(inp(n,m,0)); }

static Tensor *op_neg(const CMLONNXNode *n, TensorMap *m)
{ return uop_neg(inp(n,m,0)); }

static Tensor *op_abs(const CMLONNXNode *n, TensorMap *m)
{ return uop_abs(inp(n,m,0)); }


static Tensor *op_softmax(const CMLONNXNode *n, TensorMap *m)
{
    int axis = (int)attr_int(n, "axis", -1);
    return uop_softmax(inp(n,m,0), axis);
}


static Tensor *op_reshape(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x     = inp(n, m, 0);
    Tensor *shape = inp(n, m, 1);
    if (!x || !shape) return NULL;

    tensor_ensure_executed(shape);
    float *sdata = (float *)tensor_data_ptr(shape);
    if (!sdata) return NULL;

    int new_ndim = (int)shape->numel;
    int new_shape[8];
    int inferred_idx = -1;

    for (int i = 0; i < new_ndim && i < 8; i++) {
        new_shape[i] = (int)sdata[i];
        if (new_shape[i] == 0) {
            new_shape[i] = (i < x->ndim) ? x->shape[i] : 1;
        } else if (new_shape[i] == -1) {
            inferred_idx = i;
        }
    }

    if (inferred_idx >= 0) {
        size_t known = 1;
        for (int i = 0; i < new_ndim; i++) {
            if (i != inferred_idx) known *= (size_t)new_shape[i];
        }
        new_shape[inferred_idx] = (known > 0) ? (int)(x->numel / known) : 1;
    }

    ReshapeParams p = { .new_shape = new_shape, .new_ndim = new_ndim };
    return uop_reshape(x, &p);
}


static Tensor *op_transpose(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    int count = 0;
    const int64_t *perm_vals = attr_ints(n, "perm", &count);

    int perm[8];
    if (perm_vals && count > 0) {
        for (int i = 0; i < count && i < 8; i++) perm[i] = (int)perm_vals[i];
    } else {
        for (int i = 0; i < x->ndim; i++) perm[i] = x->ndim - 1 - i;
        count = x->ndim;
    }

    PermuteParams p = { .perm = perm, .num_dims = count };
    return uop_permute(x, &p);
}


static Tensor *op_concat(const CMLONNXNode *n, TensorMap *m)
{
    int axis = (int)attr_int(n, "axis", 0);
    Tensor *tensors[CML_ONNX_MAX_INPUTS];
    int num = 0;
    for (int i = 0; i < n->num_inputs && i < CML_ONNX_MAX_INPUTS; i++) {
        Tensor *t = inp(n, m, i);
        if (t) tensors[num++] = t;
    }
    if (num == 0) return NULL;
    return uop_cat(tensors, num, axis);
}


static Tensor *op_gemm(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *A = inp(n, m, 0);
    Tensor *B = inp(n, m, 1);
    Tensor *C = inp(n, m, 2); /* optional bias */
    if (!A || !B) return NULL;

    float alpha = attr_float(n, "alpha", 1.0f);
    float beta  = attr_float(n, "beta",  1.0f);
    int transA  = (int)attr_int(n, "transA", 0);
    int transB  = (int)attr_int(n, "transB", 0);

    /* Transpose if needed. The permuted view must be materialized into a
     * contiguous buffer before uop_matmul: feeding it a raw permute view
     * yields silently wrong results. */
    if (transA) {
        int perm[2] = {1, 0};
        PermuteParams pp = { .perm = perm, .num_dims = 2 };
        A = uop_permute(A, &pp);
        A = A ? tensor_contiguous(A) : NULL;
    }
    if (transB) {
        int perm[2] = {1, 0};
        PermuteParams pp = { .perm = perm, .num_dims = 2 };
        B = uop_permute(B, &pp);
        B = B ? tensor_contiguous(B) : NULL;
    }
    if ((transA || transB) && (!A || !B)) return NULL;
    if (!A || !B) return NULL;

    Tensor *result = uop_matmul(A, B);
    if (!result) return NULL;

    /* The alpha/beta scaling turns this into a three-level lazy chain
     * (matmul -> mul -> add), and realizing such a chain from the tail
     * currently yields zeros: the intermediate matmul is never executed.
     * Materialize it up front so downstream ops see real data. */
    tensor_ensure_executed(result);

    /* Scale by alpha if != 1 */
    if (fabsf(alpha - 1.0f) > 1e-7f) {
        Tensor *alpha_t = tensor_full((int[]){1}, 1, NULL, alpha);
        result = uop_mul(result, alpha_t);
    }

    /* Add bias scaled by beta */
    if (C) {
        if (fabsf(beta - 1.0f) > 1e-7f) {
            Tensor *beta_t = tensor_full((int[]){1}, 1, NULL, beta);
            C = uop_mul(C, beta_t);
        }
        result = uop_add(result, C);
    }

    return result;
}


static Tensor *op_conv(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    Tensor *w = inp(n, m, 1);
    Tensor *b = inp(n, m, 2); /* optional */
    if (!x || !w) return NULL;

    int kcount = 0;
    const int64_t *kernel_shape = attr_ints(n, "kernel_shape", &kcount);
    int scount = 0;
    const int64_t *strides_v = attr_ints(n, "strides", &scount);
    int pcount = 0;
    const int64_t *pads = attr_ints(n, "pads", &pcount);
    int dcount = 0;
    const int64_t *dilations = attr_ints(n, "dilations", &dcount);
    int group = (int)attr_int(n, "group", 1);

    int ks[2] = {3, 3};
    int st[2] = {1, 1};
    int pd[2] = {0, 0};
    int dl[2] = {1, 1};

    if (kernel_shape && kcount >= 2) {
        ks[0] = (int)kernel_shape[0];
        ks[1] = (int)kernel_shape[1];
    } else if (w->ndim >= 4) {
        /* Infer from weight shape: [OC, IC/g, kH, kW] */
        ks[0] = w->shape[2];
        ks[1] = w->shape[3];
    }
    if (strides_v && scount >= 2) {
        st[0] = (int)strides_v[0];
        st[1] = (int)strides_v[1];
    }
    if (pads && pcount >= 2) {
        /* ONNX pads: [top, left, bottom, right] -- we take top/left */
        pd[0] = (int)pads[0];
        pd[1] = (int)pads[1];
    }
    if (dilations && dcount >= 2) {
        dl[0] = (int)dilations[0];
        dl[1] = (int)dilations[1];
    }

    Conv2DParams params = {
        .kernel_size = ks,
        .stride      = st,
        .padding     = pd,
        .dilation    = dl,
        .groups      = group,
        .bias        = (b != NULL),
    };

    return uop_conv2d(x, w, b, &params);
}

static Tensor *op_batchnorm(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x     = inp(n, m, 0);
    Tensor *scale = inp(n, m, 1);
    Tensor *bias  = inp(n, m, 2);
    Tensor *mean  = inp(n, m, 3);
    Tensor *var   = inp(n, m, 4);
    if (!x || !scale || !bias || !mean || !var) return NULL;

    float eps = attr_float(n, "epsilon", 1e-5f);

    Tensor *eps_t = tensor_full((int[]){1}, 1, NULL, eps);
    Tensor *xm = uop_sub(x, mean);
    Tensor *ve  = uop_add(var, eps_t);
    Tensor *sve = uop_sqrt(ve);
    Tensor *norm   = uop_div(xm, sve);
    Tensor *scaled = uop_mul(norm, scale);
    return uop_add(scaled, bias);
}

static Tensor *op_maxpool(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    int kcount = 0;
    const int64_t *kernel_shape = attr_ints(n, "kernel_shape", &kcount);

    if (x->ndim == 4 && kernel_shape && kcount >= 2) {
        int reduce_dims[2] = {2, 3};
        ReduceParams rp = {
            .dims = reduce_dims,
            .num_dims = 2,
            .keepdim = true,
        };

        Tensor *pooled = uop_max_reduce(x, &rp);
        return pooled;
    }

    return x;
}

static Tensor *op_avgpool(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    if (x->ndim == 4) {
        int reduce_dims[2] = {2, 3};
        ReduceParams rp = {
            .dims = reduce_dims,
            .num_dims = 2,
            .keepdim = true,
        };
        return uop_mean(x, &rp);
    }
    return x;
}


static Tensor *op_global_avg_pool(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    int spatial = x->ndim - 2;
    if (spatial <= 0) return x;

    int dims[8];
    for (int i = 0; i < spatial && i < 8; i++) dims[i] = i + 2;

    ReduceParams rp = {
        .dims = dims,
        .num_dims = spatial,
        .keepdim = true,
    };
    return uop_mean(x, &rp);
}


static Tensor *op_flatten(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;
    int axis = (int)attr_int(n, "axis", 1);
    return uop_flatten(x, axis, x->ndim - 1);
}


static Tensor *op_squeeze(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    int acount = 0;
    const int64_t *axes = attr_ints(n, "axes", &acount);

    int new_shape[8];
    int new_ndim = 0;

    if (axes && acount > 0) {
        for (int i = 0; i < x->ndim && new_ndim < 8; i++) {
            bool squeeze = false;
            for (int j = 0; j < acount; j++) {
                int ax = (int)axes[j];
                if (ax < 0) ax += x->ndim;
                if (ax == i && x->shape[i] == 1) { squeeze = true; break; }
            }
            if (!squeeze) new_shape[new_ndim++] = x->shape[i];
        }
    } else {
        Tensor *axes_tensor = inp(n, m, 1);
        if (axes_tensor) {
            tensor_ensure_executed(axes_tensor);
            float *ad = (float *)tensor_data_ptr(axes_tensor);
            int ac = (int)axes_tensor->numel;
            for (int i = 0; i < x->ndim && new_ndim < 8; i++) {
                bool squeeze = false;
                for (int j = 0; j < ac; j++) {
                    int ax = (int)ad[j];
                    if (ax < 0) ax += x->ndim;
                    if (ax == i && x->shape[i] == 1) { squeeze = true; break; }
                }
                if (!squeeze) new_shape[new_ndim++] = x->shape[i];
            }
        } else {
            for (int i = 0; i < x->ndim && new_ndim < 8; i++) {
                if (x->shape[i] != 1) new_shape[new_ndim++] = x->shape[i];
            }
        }
    }

    if (new_ndim == 0) { new_ndim = 1; new_shape[0] = 1; }

    ReshapeParams p = { .new_shape = new_shape, .new_ndim = new_ndim };
    return uop_reshape(x, &p);
}

static Tensor *op_unsqueeze(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    int acount = 0;
    const int64_t *axes = attr_ints(n, "axes", &acount);
    /* Zero-filled: acount can exceed what the fallback paths below write when
     * neither attribute nor input tensor is present. */
    int axes_buf[8] = {0};

    if (!axes || acount == 0) {
        Tensor *axes_tensor = inp(n, m, 1);
        if (axes_tensor) {
            tensor_ensure_executed(axes_tensor);
            float *ad = (float *)tensor_data_ptr(axes_tensor);
            acount = (int)axes_tensor->numel;
            if (acount > 8) acount = 8;
            for (int i = 0; i < acount; i++) axes_buf[i] = (int)ad[i];
        }
    } else {
        if (acount > 8) acount = 8;
        for (int i = 0; i < acount; i++) axes_buf[i] = (int)axes[i];
    }

    int out_ndim = x->ndim + acount;
    if (out_ndim > 8) out_ndim = 8;

    int new_shape[8];
    for (int i = 0; i < acount; i++) {
        if (axes_buf[i] < 0) axes_buf[i] += out_ndim;
    }

    for (int i = 1; i < acount; i++) {
        int key = axes_buf[i];
        int j = i - 1;
        while (j >= 0 && axes_buf[j] > key) {
            axes_buf[j + 1] = axes_buf[j];
            j--;
        }
        axes_buf[j + 1] = key;
    }

    int xi = 0;
    int ai = 0;
    for (int i = 0; i < out_ndim; i++) {
        if (ai < acount && axes_buf[ai] == i) {
            new_shape[i] = 1;
            ai++;
        } else {
            new_shape[i] = (xi < x->ndim) ? x->shape[xi++] : 1;
        }
    }

    ReshapeParams p = { .new_shape = new_shape, .new_ndim = out_ndim };
    return uop_reshape(x, &p);
}


static Tensor *op_clip(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    float min_val = -3.4e38f;
    float max_val =  3.4e38f;

    Tensor *min_t = inp(n, m, 1);
    Tensor *max_t = inp(n, m, 2);

    if (min_t) {
        tensor_ensure_executed(min_t);
        float *d = (float *)tensor_data_ptr(min_t);
        if (d) min_val = d[0];
    } else {
        min_val = attr_float(n, "min", -3.4e38f);
    }
    if (max_t) {
        tensor_ensure_executed(max_t);
        float *d = (float *)tensor_data_ptr(max_t);
        if (d) max_val = d[0];
    } else {
        max_val = attr_float(n, "max", 3.4e38f);
    }

    return uop_clamp(x, min_val, max_val);
}


static Tensor *op_gather(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x       = inp(n, m, 0);
    Tensor *indices = inp(n, m, 1);
    if (!x || !indices) return NULL;
    int axis = (int)attr_int(n, "axis", 0);
    return uop_gather(x, indices, axis);
}


static Tensor *op_pad(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    Tensor *pads_t = inp(n, m, 1);
    float constant_value = 0.0f;

    Tensor *cv = inp(n, m, 2);
    if (cv) {
        tensor_ensure_executed(cv);
        float *d = (float *)tensor_data_ptr(cv);
        if (d) constant_value = d[0];
    }

    if (!pads_t) return x;

    tensor_ensure_executed(pads_t);
    float *pd = (float *)tensor_data_ptr(pads_t);
    if (!pd) return x;

    /* ONNX pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...] */
    int pad_ndim = (int)pads_t->numel / 2;
    if (pad_ndim > x->ndim) pad_ndim = x->ndim;

    int pad_widths[16]; /* [before_0, after_0, before_1, after_1, ...] */
    for (int i = 0; i < pad_ndim; i++) {
        pad_widths[2 * i]     = (int)pd[i];           /* begin */
        pad_widths[2 * i + 1] = (int)pd[pad_ndim + i]; /* end */
    }

    return uop_pad(x, pad_widths, pad_ndim, constant_value);
}


static Tensor *op_slice(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    Tensor *starts_t = inp(n, m, 1);
    Tensor *ends_t   = inp(n, m, 2);
    Tensor *axes_t   = inp(n, m, 3);
    Tensor *steps_t  = inp(n, m, 4);

    if (!starts_t || !ends_t) return x;

    tensor_ensure_executed(starts_t);
    tensor_ensure_executed(ends_t);
    float *starts_d = (float *)tensor_data_ptr(starts_t);
    float *ends_d   = (float *)tensor_data_ptr(ends_t);
    if (!starts_d || !ends_d) return x;

    int num_slices = (int)starts_t->numel;
    int start[8], end[8], step[8];

    for (int i = 0; i < x->ndim; i++) {
        start[i] = 0;
        end[i]   = x->shape[i];
        step[i]  = 1;
    }

    float *axes_d = NULL;
    float *steps_d = NULL;
    if (axes_t) { tensor_ensure_executed(axes_t); axes_d = (float *)tensor_data_ptr(axes_t); }
    if (steps_t) { tensor_ensure_executed(steps_t); steps_d = (float *)tensor_data_ptr(steps_t); }

    for (int i = 0; i < num_slices; i++) {
        int axis = axes_d ? (int)axes_d[i] : i;
        if (axis < 0) axis += x->ndim;
        if (axis < 0 || axis >= x->ndim) continue;

        int s = (int)starts_d[i];
        int e = (int)ends_d[i];
        int st_val = steps_d ? (int)steps_d[i] : 1;

        if (s < 0) s += x->shape[axis];
        if (e < 0) e += x->shape[axis];
        if (s < 0) s = 0;
        if (e > x->shape[axis]) e = x->shape[axis];

        start[axis] = s;
        end[axis]   = e;
        step[axis]  = st_val;
    }

    SliceParams sp = {
        .start    = start,
        .end      = end,
        .step     = step,
        .num_dims = x->ndim,
    };
    return uop_slice(x, &sp);
}


static Tensor *op_cast(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;
    int to = (int)attr_int(n, "to", 1);

    DType target;
    switch (to) {
    case 1:  target = DTYPE_FLOAT32; break;
    case 2:  target = DTYPE_UINT8;   break;
    case 3:  target = DTYPE_INT8;    break;
    case 6:  target = DTYPE_INT32;   break;
    case 7:  target = DTYPE_INT64;   break;
    case 10: target = DTYPE_FLOAT16; break;
    case 11: target = DTYPE_FLOAT64; break;
    default: target = DTYPE_FLOAT32; break;
    }

    return tensor_cast(x, target);
}


static Tensor *op_identity(const CMLONNXNode *n, TensorMap *m)
{ return inp(n, m, 0); }

static Tensor *op_dropout(const CMLONNXNode *n, TensorMap *m)
{
    return inp(n, m, 0);
}


static Tensor *op_shape(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    TensorConfig cfg = { .dtype = DTYPE_FLOAT32, .has_dtype = true };
    Tensor *out = tensor_empty((int[]){x->ndim}, 1, &cfg);
    if (!out) return NULL;

    float *d = (float *)tensor_data_ptr(out);
    if (d) {
        for (int i = 0; i < x->ndim; i++) d[i] = (float)x->shape[i];
    }
    return out;
}


static Tensor *op_constant(const CMLONNXNode *n, TensorMap *m)
{
    (void)m;

    const CMLONNXAttribute *va = find_attr(n, "value");
    if (va && va->type == CML_ONNX_ATTR_TENSOR && va->value.tensor) {
        return va->value.tensor;
    }

    const CMLONNXAttribute *vf = find_attr(n, "value_float");
    if (vf && vf->type == CML_ONNX_ATTR_FLOAT) {
        return tensor_full((int[]){1}, 1, NULL, vf->value.f);
    }

    const CMLONNXAttribute *vi = find_attr(n, "value_int");
    if (vi && vi->type == CML_ONNX_ATTR_INT) {
        return tensor_full((int[]){1}, 1, NULL, (float)vi->value.i);
    }

    const CMLONNXAttribute *vfs = find_attr(n, "value_floats");
    if (vfs && vfs->type == CML_ONNX_ATTR_FLOATS && vfs->value.floats.count > 0) {
        int shape[1] = { vfs->value.floats.count };
        return tensor_from_data(vfs->value.floats.data, shape, 1, NULL);
    }

    const CMLONNXAttribute *vis = find_attr(n, "value_ints");
    if (vis && vis->type == CML_ONNX_ATTR_INTS && vis->value.ints.count > 0) {
        int count = vis->value.ints.count;
        float *fdata = (float *)cml_malloc(sizeof(float) * (size_t)count);
        if (!fdata) return NULL;
        for (int i = 0; i < count; i++) fdata[i] = (float)vis->value.ints.data[i];
        int shape[1] = { count };
        Tensor *t = tensor_from_data(fdata, shape, 1, NULL);
        cml_free(fdata);
        return t;
    }

    LOG_WARNING("onnx_ops: Constant node '%s' has no recognised value attribute", n->name);
    return tensor_full((int[]){1}, 1, NULL, 0.0f);
}


static Tensor *op_where(const CMLONNXNode *n, TensorMap *m)
{
    WhereParams p = {
        .cond = inp(n, m, 0),
        .a    = inp(n, m, 1),
        .b    = inp(n, m, 2),
    };
    return uop_where(&p);
}

static Tensor *op_expand(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x       = inp(n, m, 0);
    Tensor *shape_t = inp(n, m, 1);
    if (!x || !shape_t) return NULL;

    int count = 0;
    const float *sd = tensor_floats(shape_t, &count);
    if (!sd || count < 1 || count > 8) return NULL;

    /* Unidirectional (right-aligned) broadcast: lift the input to the output
     * rank first so every output dim maps onto an input dim. */
    int shift = count - x->ndim;
    if (shift > 0) {
        int shp[8];
        for (int i = 0; i < count; i++)
            shp[i] = (i - shift >= 0) ? x->shape[i - shift] : 1;
        x = uop_reshape_to(x, shp, count);
        if (!x) return NULL;
    } else if (shift < 0) {
        return NULL; /* ONNX forbids lowering the rank */
    }

    int shape[8];
    for (int i = 0; i < count; i++) {
        int d = (int)sd[i];
        if (d <= 0) d = x->shape[i];
        shape[i] = d > 0 ? d : 1;
    }

    return uop_expand_to(x, shape, count);
}

static Tensor *op_tile(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x || x->ndim > 8) return NULL;

    int rcount = 0;
    const float *rd = tensor_floats(inp(n, m, 1), &rcount);
    if (!rd || rcount != x->ndim) return NULL;

    int out_shape[8];
    size_t x_strides[8], out_numel = 1;
    for (int i = 0; i < x->ndim; i++) {
        out_shape[i] = x->shape[i] * (int)rd[i];
        out_numel *= (size_t)out_shape[i];
    }
    x_strides[x->ndim - 1] = 1;
    for (int i = x->ndim - 2; i >= 0; i--)
        x_strides[i] = x_strides[i + 1] * (size_t)x->shape[i + 1];

    const float *xd = (const float *)tensor_data_ptr(x);

    float *out = (float *)cml_malloc(sizeof(float) * out_numel);
    if (!out) return NULL;

    int idx[8] = {0};
    for (size_t o = 0; o < out_numel; o++) {
        size_t src = 0;
        for (int i = 0; i < x->ndim; i++)
            src += (size_t)(idx[i] % x->shape[i]) * x_strides[i];
        out[o] = xd[src];

        for (int i = x->ndim - 1; i >= 0; i--) {
            if (++idx[i] < out_shape[i]) break;
            idx[i] = 0;
        }
    }

    TensorConfig cfg = { .dtype = DTYPE_FLOAT32, .has_dtype = true };
    Tensor *result = tensor_from_data(out, out_shape, x->ndim, &cfg);
    cml_free(out);
    return result;
}

static Tensor *op_range(const CMLONNXNode *n, TensorMap *m)
{
    int scount = 0, lcount = 0, dcount = 0;
    const float *s = tensor_floats(inp(n, m, 0), &scount);
    const float *l = tensor_floats(inp(n, m, 1), &lcount);
    const float *d = tensor_floats(inp(n, m, 2), &dcount);
    if (!s || !l || !d || dcount == 0 || d[0] == 0.0f) return NULL;

    int count = (int)ceilf((l[0] - s[0]) / d[0]);
    if (count < 0) count = 0;

    int shape[1] = { count };
    TensorConfig cfg = { .dtype = DTYPE_FLOAT32, .has_dtype = true };
    Tensor *out = tensor_empty(shape, 1, &cfg);
    if (!out) return NULL;

    float *od = (float *)tensor_data_ptr(out);
    for (int i = 0; i < count; i++) od[i] = s[0] + d[0] * (float)i;
    return out;
}

static Tensor *op_cumsum(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    int exclusive = attr_int(n, "exclusive", 0);
    int reverse   = attr_int(n, "reverse", 0);

    int acount = 0;
    const float *ad = tensor_floats(inp(n, m, 1), &acount);
    if (!ad || acount < 1) return NULL;
    int axis = (int)ad[0];

    /* Compose the exclusive/reverse variants from the inclusive scan:
     *   reverse   -> flip, scan, flip back
     *   exclusive -> inclusive scan minus the element itself (sum_{j<i}) */
    Tensor *y = reverse ? tensor_flip(x, axis) : x;
    if (!y) return NULL;
    Tensor *c = uop_cumsum(y, axis);
    if (!c) return NULL;
    if (exclusive) {
        c = uop_sub(c, y);
        if (!c) return NULL;
    }
    return reverse ? tensor_flip(c, axis) : c;
}

static Tensor *op_scatter_nd(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *data = inp(n, m, 0);
    Tensor *idx  = inp(n, m, 1);
    Tensor *upd  = inp(n, m, 2);
    if (!data || !idx || !upd || data->ndim > 8) return NULL;

    int k = idx->ndim;
    int q = (k >= 1) ? idx->shape[k - 1] : 0;
    if (q < 1 || q > data->ndim) return NULL;

    size_t outer = 1; /* number of indexed slices */
    for (int i = 0; i < k - 1; i++) outer *= (size_t)idx->shape[i];

    size_t slice_elems = 1; /* elements per indexed slice */
    for (int i = q; i < data->ndim; i++) slice_elems *= (size_t)data->shape[i];

    int out_shape[8];
    size_t d_strides[8], numel = 1;
    for (int i = data->ndim - 1; i >= 0; i--) {
        out_shape[i]    = data->shape[i];
        d_strides[i]    = (i == data->ndim - 1) ? 1 : d_strides[i + 1] * (size_t)data->shape[i + 1];
        numel           *= (size_t)data->shape[i];
    }

    const float *dd = (const float *)tensor_data_ptr(data);
    const float *id = (const float *)tensor_data_ptr(idx);
    const float *ud = (const float *)tensor_data_ptr(upd);

    float *out = (float *)cml_malloc(sizeof(float) * numel);
    if (!out) return NULL;
    memcpy(out, dd, sizeof(float) * numel);

    for (size_t e = 0; e < outer; e++) {
        size_t base = 0;
        for (int t = 0; t < q; t++) {
            int ix = (int)id[e * (size_t)q + (size_t)t];
            if (ix < 0) ix += data->shape[t]; /* negative indices wrap */
            base += (size_t)ix * d_strides[t];
        }
        memcpy(out + base, ud + e * slice_elems,
               sizeof(float) * slice_elems);
    }

    TensorConfig cfg = { .dtype = DTYPE_FLOAT32, .has_dtype = true };
    Tensor *result = tensor_from_data(out, out_shape, data->ndim, &cfg);
    cml_free(out);
    return result;
}

typedef enum {
    RESIZE_COORD_HALF_PIXEL = 0,
    RESIZE_COORD_ASYMMETRIC,
    RESIZE_COORD_ALIGN_CORNERS,
    RESIZE_COORD_PYTORCH_HALF_PIXEL,
} ResizeCoordMode;

static void resize_sample_coord(int in_size, int out_size, bool linear,
                                ResizeCoordMode coord, const char *nearest_mode,
                                int o, int *i0, int *i1, float *w0, float *w1)
{
    float scale = (float)in_size / (float)out_size;
    float u;
    switch (coord) {
    case RESIZE_COORD_ASYMMETRIC:
        u = (float)o * scale;
        break;
    case RESIZE_COORD_ALIGN_CORNERS:
        /* Endpoints coincide; out_size==1 collapses to the first sample. */
        u = (out_size > 1)
                ? (float)o * (float)(in_size - 1) / (float)(out_size - 1)
                : 0.0f;
        break;
    case RESIZE_COORD_PYTORCH_HALF_PIXEL:
        u = (out_size > 1) ? ((float)o + 0.5f) * scale - 0.5f : 0.0f;
        break;
    case RESIZE_COORD_HALF_PIXEL:
    default:
        u = ((float)o + 0.5f) * scale - 0.5f;
        break;
    }

    if (!linear) {
        int v;
        if (strcmp(nearest_mode, "floor") == 0)          v = (int)floorf(u);
        else if (strcmp(nearest_mode, "ceil") == 0)      v = (int)ceilf(u);
        else if (strcmp(nearest_mode, "round_prefer_ceil") == 0)
                                                         v = (int)ceilf(u - 0.5f);
        else /* round_prefer_floor (ONNX default) */     v = (int)floorf(u + 0.5f);
        if (v < 0) v = 0;
        if (v > in_size - 1) v = in_size - 1;
        *i0 = v; *i1 = v; *w0 = 1.0f; *w1 = 0.0f;
        return;
    }

    if (u < 0.0f) u = 0.0f;
    if (u > (float)(in_size - 1)) u = (float)(in_size - 1);
    int lo = (int)floorf(u);
    if (lo > in_size - 1) lo = in_size - 1;
    *i0 = lo;
    *i1 = (lo + 1 < in_size) ? lo + 1 : lo;
    *w1 = u - (float)lo;
    *w0 = 1.0f - *w1;
}

static Tensor *op_resize(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x || x->ndim < 1 || x->ndim > 8) return NULL;

    char mode[16]        = "nearest";
    char coord_mode[16]  = "half_pixel";
    char nearest_mode[24] = "round_prefer_floor";
    attr_string(n, "mode", mode, sizeof(mode));
    attr_string(n, "coordinate_transformation_mode", coord_mode, sizeof(coord_mode));
    attr_string(n, "nearest_mode", nearest_mode, sizeof(nearest_mode));

    bool linear = (strcmp(mode, "linear") == 0);
    if (!linear && strcmp(mode, "nearest") != 0) {
        LOG_ERROR("onnx_ops: Resize mode '%s' not supported by importer", mode);
        return NULL;
    }
    ResizeCoordMode coord;
    if (strcmp(coord_mode, "half_pixel") == 0)            coord = RESIZE_COORD_HALF_PIXEL;
    else if (strcmp(coord_mode, "asymmetric") == 0)       coord = RESIZE_COORD_ASYMMETRIC;
    else if (strcmp(coord_mode, "align_corners") == 0)    coord = RESIZE_COORD_ALIGN_CORNERS;
    else if (strcmp(coord_mode, "pytorch_half_pixel") == 0)
                                                          coord = RESIZE_COORD_PYTORCH_HALF_PIXEL;
    else {
        LOG_ERROR("onnx_ops: Resize coordinate mode '%s' not supported by importer",
                  coord_mode);
        return NULL;
    }

    int zcount = 0, scount = 0;
    const float *sizes  = tensor_floats(inp(n, m, 3), &zcount);
    const float *scales = tensor_floats(inp(n, m, 2), &scount);

    int out_shape[8];
    size_t out_numel = 1;
    for (int i = 0; i < x->ndim; i++) {
        int od;
        if (sizes && zcount == x->ndim) {
            od = (int)sizes[i];
        } else if (scales && scount == x->ndim) {
            od = (int)floorf((float)x->shape[i] * scales[i]);
        } else {
            LOG_ERROR("onnx_ops: Resize needs valid 'scales' or 'sizes' input");
            return NULL;
        }
        if (od < 1) od = 1;
        out_shape[i] = od;
        out_numel   *= (size_t)od;
    }

    float *out = (float *)cml_malloc(sizeof(float) * out_numel);
    if (!out) return NULL;

    size_t strides[8];
    strides[x->ndim - 1] = 1;
    for (int d = x->ndim - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * (size_t)x->shape[d + 1];

    const float *xd = (const float *)tensor_data_ptr(x);

    int cur[8] = {0};
    for (size_t o = 0; o < out_numel; o++) {
        int   i0[8], i1[8], two[8], ntwo = 0;
        float w0[8], w1[8];
        for (int d = 0; d < x->ndim; d++) {
            resize_sample_coord(x->shape[d], out_shape[d], linear, coord,
                                nearest_mode, cur[d],
                                &i0[d], &i1[d], &w0[d], &w1[d]);
            if (i1[d] != i0[d] && w1[d] > 0.0f) two[ntwo++] = d;
        }

        /* Linear interpolation sums over the corners that actually vary;
         * every other dim contributes a single fixed index. */
        float val = 0.0f;
        for (int mask = 0; mask < (1 << ntwo); mask++) {
            float w = 1.0f;
            size_t off = 0;
            for (int b = 0; b < ntwo; b++) {
                int d = two[b];
                w *= ((mask >> b) & 1) ? w1[d] : w0[d];
            }
            for (int d = 0; d < x->ndim; d++) {
                int bit = -1;
                for (int b = 0; b < ntwo; b++)
                    if (two[b] == d) { bit = b; break; }
                bool upper = (bit >= 0) && ((mask >> bit) & 1);
                off += (size_t)(upper ? i1[d] : i0[d]) * strides[d];
            }
            val += w * xd[off];
        }
        out[o] = val;

        for (int i = x->ndim - 1; i >= 0; i--) {
            if (++cur[i] < out_shape[i]) break;
            cur[i] = 0;
        }
    }

    TensorConfig cfg = { .dtype = DTYPE_FLOAT32, .has_dtype = true };
    Tensor *result = tensor_from_data(out, out_shape, x->ndim, &cfg);
    cml_free(out);
    return result;
}

/* Reduce-family axes: opset < 13 carried them in the "axes" attribute,
 * newer opsets pass them as an optional second input. */
static int reduce_axes(const CMLONNXNode *n, TensorMap *m, int ndim, int *axes)
{
    int count = 0;
    Tensor *axes_t = inp(n, m, 1);
    if (axes_t) {
        int acount = 0;
        const float *ad = tensor_floats(axes_t, &acount);
        if (!ad) return -1;
        for (int i = 0; i < acount && count < 8; i++)
            axes[count++] = (int)ad[i];
    } else {
        int acount = 0;
        const int64_t *av = attr_ints(n, "axes", &acount);
        for (int i = 0; av && i < acount && count < 8; i++)
            axes[count++] = (int)av[i];
    }

    for (int i = 0; i < count; i++)
        if (axes[i] < 0) axes[i] += ndim;
    return count;
}

typedef Tensor *(*reduce_fn)(Tensor *, ReduceParams *);

static Tensor *run_reduction(const CMLONNXNode *n, TensorMap *m, reduce_fn fn)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    int axes[8];
    int count = reduce_axes(n, m, x->ndim, axes);
    if (count < 0) return NULL;

    if (count == 0) {
        if (attr_int(n, "noop_with_empty_axes", 0) != 0) return x;
        for (int i = 0; i < x->ndim; i++) axes[i] = i;
        count = x->ndim;
    }

    ReduceParams rp = {
        .dims     = axes,
        .num_dims = count,
        .keepdim  = attr_int(n, "keepdim", 1) != 0,
    };
    return fn(x, &rp);
}

static Tensor *op_reduce_min(const CMLONNXNode *n, TensorMap *m)
{ return run_reduction(n, m, uop_min_reduce); }

static Tensor *op_reduce_max(const CMLONNXNode *n, TensorMap *m)
{ return run_reduction(n, m, uop_max_reduce); }

static Tensor *op_reduce_prod(const CMLONNXNode *n, TensorMap *m)
{ return run_reduction(n, m, uop_prod); }

static Tensor *op_reduce_sum(const CMLONNXNode *n, TensorMap *m)
{ return run_reduction(n, m, uop_sum); }

static Tensor *op_reduce_mean(const CMLONNXNode *n, TensorMap *m)
{ return run_reduction(n, m, uop_mean); }

static Tensor *arg_reduce(const CMLONNXNode *n, TensorMap *m,
                          reduce_fn fn)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    int axis = (int)attr_int(n, "axis", 0);
    if (axis < 0) axis += x->ndim;

    ReduceParams rp = {
        .dims     = &axis,
        .num_dims = 1,
        .keepdim  = attr_int(n, "keepdim", 1) != 0,
    };
    return fn(x, &rp);
}

static Tensor *op_argmin(const CMLONNXNode *n, TensorMap *m)
{ return arg_reduce(n, m, uop_argmin); }

static Tensor *op_argmax(const CMLONNXNode *n, TensorMap *m)
{ return arg_reduce(n, m, uop_argmax); }

static Tensor *op_erf(const CMLONNXNode *n, TensorMap *m)
{ return uop_erf(inp(n, m, 0)); }

static Tensor *op_leaky_relu(const CMLONNXNode *n, TensorMap *m)
{
    return uop_leaky_relu(inp(n, m, 0), attr_float(n, "alpha", 0.01f));
}

/* PReLU(x, slope) = max(x, 0) + slope * min(x, 0). The slope broadcasts
 * against x with its single axis on dim 1 (the channel axis). */
static Tensor *op_prelu(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x     = inp(n, m, 0);
    Tensor *slope = inp(n, m, 1);
    if (!x || !slope) return NULL;

    Tensor *s = slope;
    if (s->ndim != x->ndim) {
        int shp[8];
        for (int i = 0; i < x->ndim; i++) shp[i] = 1;
        int ax = (x->ndim > 1) ? 1 : 0;
        shp[ax] = (int)s->numel;
        s = uop_reshape_to(s, shp, x->ndim);
        if (!s) return NULL;
    }

    Tensor *zero = tensor_full(x->shape, x->ndim, NULL, 0.0f);
    if (!zero) return NULL;

    Tensor *pos = uop_max(x, zero);
    Tensor *neg = uop_minimum(x, zero);
    if (!pos || !neg) return NULL;

    Tensor *sneg = uop_mul(s, neg);
    if (!sneg) return NULL;
    return uop_add(pos, sneg);
}

static Tensor *op_softplus(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x) return NULL;

    Tensor *ex = uop_exp(x);
    if (!ex) return NULL;
    Tensor *one = tensor_full(ex->shape, ex->ndim, NULL, 1.0f);
    if (!one) return NULL;
    Tensor *denom = uop_add(ex, one);
    if (!denom) return NULL;
    return uop_log(denom);
}

static Tensor *split_piece(Tensor *x, int axis, int start, int end)
{
    int startv[8], endv[8], stepv[8];
    for (int i = 0; i < x->ndim; i++) {
        startv[i] = 0;
        endv[i]   = x->shape[i];
        stepv[i]  = 1;
    }
    startv[axis] = start;
    endv[axis]   = end;

    SliceParams sp = {
        .start    = startv,
        .end      = endv,
        .step     = stepv,
        .num_dims = x->ndim,
    };
    return uop_slice(x, &sp);
}

/* Multi-output: registers every piece in the tensor map itself and returns
 * output 0; cml_onnx_run skips its blanket single-result assignment when a
 * node declares more than one output. */
static Tensor *op_split(const CMLONNXNode *n, TensorMap *m)
{
    Tensor *x = inp(n, m, 0);
    if (!x || n->num_outputs < 1) return NULL;

    int axis = (int)attr_int(n, "axis", 0);
    if (axis < 0) axis += x->ndim;
    if (axis < 0 || axis >= x->ndim) return NULL;

    int num_out = n->num_outputs;
    int sizes[CML_ONNX_MAX_OUTPUTS] = {0};

    int scount = 0;
    const float *sd = tensor_floats(inp(n, m, 1), &scount);
    if (sd && scount == num_out) {
        for (int i = 0; i < num_out; i++) sizes[i] = (int)sd[i];
    } else {
        int chunk = (x->shape[axis] + num_out - 1) / num_out; /* ceil split */
        int remaining = x->shape[axis];
        for (int i = 0; i < num_out; i++) {
            sizes[i] = (chunk < remaining) ? chunk : remaining;
            if (sizes[i] < 0) sizes[i] = 0;
            remaining -= sizes[i];
        }
    }

    Tensor *first = NULL;
    int start = 0;
    for (int i = 0; i < num_out; i++) {
        int end = start + sizes[i];
        Tensor *piece = split_piece(x, axis, start, end);
        if (!piece) return NULL;
        if (!first) first = piece;
        if (n->outputs[i])
            tensor_map_set(m, n->outputs[i], piece);
        start = end;
    }
    return first;
}

static Tensor *op_pow(const CMLONNXNode *n, TensorMap *m)
{ return uop_pow(inp(n, m, 0), inp(n, m, 1)); }

static Tensor *op_reciprocal(const CMLONNXNode *n, TensorMap *m)
{ return uop_recip(inp(n, m, 0)); }

static Tensor *op_floor(const CMLONNXNode *n, TensorMap *m)
{ return uop_floor(inp(n, m, 0)); }

static Tensor *op_ceil(const CMLONNXNode *n, TensorMap *m)
{ return uop_ceil(inp(n, m, 0)); }

static Tensor *op_round(const CMLONNXNode *n, TensorMap *m)
{ return uop_round(inp(n, m, 0)); }

static Tensor *op_sign(const CMLONNXNode *n, TensorMap *m)
{ return uop_sign(inp(n, m, 0)); }

typedef Tensor *(*binary_fn)(Tensor *, Tensor *);

static Tensor *fold_binary(const CMLONNXNode *n, TensorMap *m, binary_fn bin)
{
    Tensor *acc = inp(n, m, 0);
    if (!acc) return NULL;
    for (int i = 1; i < n->num_inputs; i++) {
        Tensor *t = inp(n, m, i);
        if (!t) continue;
        acc = bin(acc, t);
        if (!acc) return NULL;
    }
    return acc;
}

static Tensor *op_elw_min(const CMLONNXNode *n, TensorMap *m)
{ return fold_binary(n, m, uop_minimum); }

static Tensor *op_elw_max(const CMLONNXNode *n, TensorMap *m)
{ return fold_binary(n, m, uop_max); }

static Tensor *op_sum_all(const CMLONNXNode *n, TensorMap *m)
{ return fold_binary(n, m, uop_add); }

static Tensor *op_elw_mean(const CMLONNXNode *n, TensorMap *m)
{
    int num = 0;
    for (int i = 0; i < n->num_inputs; i++)
        if (inp(n, m, i)) num++;
    Tensor *acc = fold_binary(n, m, uop_add);
    if (!acc || num <= 1) return acc;
    return uop_div(acc, tensor_full((int[]){1}, 1, NULL, (float)num));
}

typedef struct {
    const char *name;
    onnx_op_fn  fn;
} OnnxOpEntry;

static const OnnxOpEntry g_op_table[] = {
    { "Add",                  op_add            },
    { "Sub",                  op_sub            },
    { "Mul",                  op_mul            },
    { "Div",                  op_div            },
    { "MatMul",               op_matmul         },
    { "Relu",                 op_relu           },
    { "Sigmoid",              op_sigmoid        },
    { "Tanh",                 op_tanh           },
    { "Exp",                  op_exp            },
    { "Log",                  op_log            },
    { "Sqrt",                 op_sqrt           },
    { "Neg",                  op_neg            },
    { "Abs",                  op_abs            },
    { "Softmax",              op_softmax        },
    { "Reshape",              op_reshape        },
    { "Transpose",            op_transpose      },
    { "Concat",               op_concat         },
    { "Gemm",                 op_gemm           },
    { "Conv",                 op_conv           },
    { "BatchNormalization",   op_batchnorm      },
    { "MaxPool",              op_maxpool        },
    { "AveragePool",          op_avgpool        },
    { "GlobalAveragePool",    op_global_avg_pool},
    { "Flatten",              op_flatten        },
    { "Squeeze",              op_squeeze        },
    { "Unsqueeze",            op_unsqueeze      },
    { "Clip",                 op_clip           },
    { "Gather",               op_gather         },
    { "Pad",                  op_pad            },
    { "Slice",                op_slice          },
    { "Cast",                 op_cast           },
    { "Identity",             op_identity       },
    { "Dropout",              op_dropout        },
    { "Shape",                op_shape          },
    { "Constant",             op_constant       },
    { "Where",                op_where          },
    { "Expand",               op_expand         },
    { "Tile",                 op_tile           },
    { "Range",                op_range          },
    { "CumSum",               op_cumsum         },
    { "ScatterND",            op_scatter_nd     },
    { "Resize",               op_resize         },
    { "ReduceMin",            op_reduce_min     },
    { "ReduceMax",            op_reduce_max     },
    { "ReduceProd",           op_reduce_prod    },
    { "ReduceSum",            op_reduce_sum     },
    { "ReduceMean",           op_reduce_mean    },
    { "ArgMin",               op_argmin         },
    { "ArgMax",               op_argmax         },
    { "Erf",                  op_erf            },
    { "LeakyRelu",            op_leaky_relu     },
    { "PReLU",                op_prelu          },
    { "Softplus",             op_softplus       },
    { "Split",                op_split          },
    { "Pow",                  op_pow            },
    { "Reciprocal",           op_reciprocal     },
    { "Floor",                op_floor          },
    { "Ceil",                 op_ceil           },
    { "Round",                op_round          },
    { "Sign",                 op_sign           },
    { "Min",                  op_elw_min        },
    { "Max",                  op_elw_max        },
    { "Sum",                  op_sum_all        },
    { "Mean",                 op_elw_mean       },
};

#define NUM_SUPPORTED_OPS ((int)(sizeof(g_op_table) / sizeof(g_op_table[0])))

bool cml_onnx_op_supported(const char *op_type)
{
    if (!op_type) return false;
    for (int i = 0; i < NUM_SUPPORTED_OPS; i++) {
        if (strcmp(g_op_table[i].name, op_type) == 0) return true;
    }
    return false;
}

static onnx_op_fn find_op_handler(const char *op_type)
{
    for (int i = 0; i < NUM_SUPPORTED_OPS; i++) {
        if (strcmp(g_op_table[i].name, op_type) == 0) return g_op_table[i].fn;
    }
    return NULL;
}

int cml_onnx_run(CMLONNXModel *model, Tensor **inputs, int num_inputs,
                 Tensor **outputs, int num_outputs)
{
    if (!model || !inputs || !outputs) return -1;

    CMLONNXGraph *g = &model->graph;

    TensorMap *map = (TensorMap *)cml_calloc(1, sizeof(TensorMap));
    if (!map) return -1;

    for (int i = 0; i < g->num_initializers; i++) {
        if (g->initializers[i].name[0] && g->initializers[i].tensor) {
            tensor_map_set(map, g->initializers[i].name,
                           g->initializers[i].tensor);
        }
    }

    /* Skip inputs that are also initializers (ONNX convention) */
    int input_idx = 0;
    for (int i = 0; i < g->num_inputs && input_idx < num_inputs; i++) {
        const char *iname = g->inputs[i].name;
        if (tensor_map_get(map, iname) != NULL) continue;
        tensor_map_set(map, iname, inputs[input_idx]);
        input_idx++;
    }

    for (int i = 0; i < g->num_nodes; i++) {
        CMLONNXNode *node = &g->nodes[i];

        onnx_op_fn handler = find_op_handler(node->op_type);
        if (!handler) {
            LOG_ERROR("onnx_ops: unsupported op '%s' (node '%s')",
                      node->op_type, node->name);
            cml_free(map);
            return -2;
        }

        Tensor *result = handler(node, map);
        if (!result) {
            LOG_ERROR("onnx_ops: op '%s' (node '%s') returned NULL",
                      node->op_type, node->name);
            cml_free(map);
            return -3;
        }

        /* Multi-output handlers (Split) register every output themselves;
         * assigning `result` to all names would clobber the siblings with
         * output 0. */
        if (node->num_outputs <= 1) {
            for (int j = 0; j < node->num_outputs; j++) {
                if (node->outputs[j]) {
                    tensor_map_set(map, node->outputs[j], result);
                }
            }
        }
    }

    int copied = 0;
    for (int i = 0; i < g->num_outputs && i < num_outputs; i++) {
        Tensor *t = tensor_map_get(map, g->outputs[i].name);
        outputs[i] = t;
        if (t) copied++;
    }

    cml_free(map);

    if (copied == 0 && num_outputs > 0) {
        LOG_ERROR("onnx_ops: no graph outputs were produced");
        return -4;
    }

    return 0;
}

int cml_onnx_list_supported_ops(const char ***ops_out, int *count_out)
{
    if (!ops_out || !count_out) return -1;

    static const char *names[NUM_SUPPORTED_OPS];
    static bool initialised = false;

    if (!initialised) {
        for (int i = 0; i < NUM_SUPPORTED_OPS; i++) {
            names[i] = g_op_table[i].name;
        }
        initialised = true;
    }

    *ops_out   = names;
    *count_out = NUM_SUPPORTED_OPS;
    return 0;
}
