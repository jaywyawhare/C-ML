#include "core/onnx.h"
#include "core/logging.h"
#include "ops/uops.h"
#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

typedef Tensor *(*onnx_op_fn)(const CMLONNXNode *node, TensorMap *map);

static Tensor *inp(const CMLONNXNode *node, TensorMap *map, int idx)
{
    if (idx < 0 || idx >= node->num_inputs) return NULL;
    if (!node->inputs[idx] || node->inputs[idx][0] == '\0') return NULL;
    return tensor_map_get(map, node->inputs[idx]);
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

    /* Transpose if needed */
    if (transA) {
        int perm[2] = {1, 0};
        PermuteParams pp = { .perm = perm, .num_dims = 2 };
        A = uop_permute(A, &pp);
    }
    if (transB) {
        int perm[2] = {1, 0};
        PermuteParams pp = { .perm = perm, .num_dims = 2 };
        B = uop_permute(B, &pp);
    }

    Tensor *result = uop_matmul(A, B);
    if (!result) return NULL;

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
    int axes_buf[8];

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
        float *fdata = (float *)malloc(sizeof(float) * (size_t)count);
        if (!fdata) return NULL;
        for (int i = 0; i < count; i++) fdata[i] = (float)vis->value.ints.data[i];
        int shape[1] = { count };
        Tensor *t = tensor_from_data(fdata, shape, 1, NULL);
        free(fdata);
        return t;
    }

    LOG_WARNING("onnx_ops: Constant node '%s' has no recognised value attribute", n->name);
    return tensor_full((int[]){1}, 1, NULL, 0.0f);
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

    TensorMap *map = (TensorMap *)calloc(1, sizeof(TensorMap));
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
            free(map);
            return -2;
        }

        Tensor *result = handler(node, map);
        if (!result) {
            LOG_ERROR("onnx_ops: op '%s' (node '%s') returned NULL",
                      node->op_type, node->name);
            free(map);
            return -3;
        }

        for (int j = 0; j < node->num_outputs; j++) {
            if (node->outputs[j]) {
                tensor_map_set(map, node->outputs[j], result);
            }
        }
    }

    int copied = 0;
    for (int i = 0; i < g->num_outputs && i < num_outputs; i++) {
        Tensor *t = tensor_map_get(map, g->outputs[i].name);
        outputs[i] = t;
        if (t) copied++;
    }

    free(map);

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
