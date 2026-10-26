#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "alloc/cml_allocator.h"

/**
 * Element-wise select: `condition` ? `x` : `y` (lazy, via uop_where).
 * @param condition (Tensor*)
 * @param x (Tensor*)
 * @param y (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_where(Tensor* condition, Tensor* x, Tensor* y) {
    if (!condition || !x || !y)
        return NULL;

    WhereParams params = {.cond = condition, .a = x, .b = y};
    return uop_where(&params);
}

/**
 * One-hot encode integer `indices` into a trailing `num_classes` dimension (lazy).
 * @param indices (Tensor*)
 * @param num_classes (int)
 * @return (Tensor*)
 */
Tensor* tensor_one_hot(Tensor* indices, int num_classes) {
    if (!indices)
        return NULL;
    return uop_one_hot(indices, num_classes);
}

/**
 * Circularly shift elements by `shift` along `axis` (lazy, via uop_roll).
 * @param t (Tensor*)
 * @param shift (int)
 * @param axis (int)
 * @return (Tensor*)
 */
Tensor* tensor_roll(Tensor* t, int shift, int axis) {
    if (!t)
        return NULL;
    return uop_roll(t, shift, axis);
}

/**
 * Return the coordinates of the non-zero elements of `t` (lazy, via uop_nonzero).
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_nonzero(Tensor* t) {
    if (!t)
        return NULL;
    return uop_nonzero(t);
}

/**
 * Element-wise copysign: magnitude of `a` with the sign of `b` (lazy).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_copysign(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_copysign(a, b);
}

/**
 * Element-wise log(exp(a) + exp(b)), computed stably (lazy, via uop_logaddexp).
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_logaddexp(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_logaddexp(a, b);
}

/**
 * Draw `num_samples` category indices from 1-D or batched 2-D `probs` (eager,
 * INT32 output). Without replacement requires num_samples <= categories.
 * Returns NULL on bad args or zero-sum probabilities.
 * @param probs (Tensor*)
 * @param num_samples (int)
 * @param replacement (bool)
 * @return (Tensor*)
 */
Tensor* tensor_multinomial(Tensor* probs, int num_samples, bool replacement) {
    if (!probs)
        return NULL;
    if (probs->ndim != 1 && probs->ndim != 2) {
        LOG_ERROR("tensor_multinomial: probs must be 1D or 2D");
        return NULL;
    }

    tensor_ensure_executed(probs);
    if (!probs->data)
        return NULL;

    bool batched       = (probs->ndim == 2);
    int batch_size     = batched ? probs->shape[0] : 1;
    int num_categories = batched ? probs->shape[1] : probs->shape[0];

    if (!replacement && num_samples > num_categories) {
        LOG_ERROR("tensor_multinomial: num_samples > num_categories without replacement");
        return NULL;
    }

    int out_shape[2];
    int out_ndim;
    if (batched) {
        out_shape[0] = batch_size;
        out_shape[1] = num_samples;
        out_ndim     = 2;
    } else {
        out_shape[0] = num_samples;
        out_ndim     = 1;
    }

    TensorConfig config = {
        .dtype = DTYPE_INT32, .device = probs->device, .has_dtype = true, .has_device = true};
    Tensor* output = tensor_empty(out_shape, out_ndim, &config);
    if (!output)
        return NULL;

    float* cumsum = (float*)cml_malloc(num_categories * sizeof(float));
    if (!cumsum) {
        tensor_free(output);
        return NULL;
    }

    int32_t* out_data = (int32_t*)tensor_data_ptr(output);

    for (int b = 0; b < batch_size; b++) {
        float total = 0.0f;
        for (int c = 0; c < num_categories; c++) {
            size_t idx = batched ? (size_t)b * num_categories + c : (size_t)c;
            total += tensor_get_float(probs, idx);
            cumsum[c] = total;
        }

        if (total <= 0.0f) {
            LOG_ERROR("tensor_multinomial: probabilities sum to zero");
            cml_free(cumsum);
            tensor_free(output);
            return NULL;
        }

        for (int c = 0; c < num_categories; c++)
            cumsum[c] /= total;

        bool* used = NULL;
        if (!replacement) {
            used = (bool*)cml_calloc(num_categories, sizeof(bool));
            if (!used) {
                cml_free(cumsum);
                tensor_free(output);
                return NULL;
            }
        }

        for (int s = 0; s < num_samples; s++) {
            float u      = (float)rand() / (float)RAND_MAX;
            int selected = num_categories - 1;

            if (!replacement && used) {
                float remaining = 0.0f;
                for (int c = 0; c < num_categories; c++) {
                    if (!used[c]) {
                        size_t idx = batched ? (size_t)b * num_categories + c : (size_t)c;
                        remaining += tensor_get_float(probs, idx);
                    }
                }
                float target = u * remaining;
                float acc    = 0.0f;
                for (int c = 0; c < num_categories; c++) {
                    if (used[c])
                        continue;
                    size_t idx = batched ? (size_t)b * num_categories + c : (size_t)c;
                    acc += tensor_get_float(probs, idx);
                    if (acc >= target) {
                        selected = c;
                        break;
                    }
                }
                used[selected] = true;
            } else {
                for (int c = 0; c < num_categories; c++) {
                    if (u <= cumsum[c]) {
                        selected = c;
                        break;
                    }
                }
            }

            size_t out_idx    = batched ? (size_t)b * num_samples + s : (size_t)s;
            out_data[out_idx] = selected;
        }

        cml_free(used);
    }

    cml_free(cumsum);
    return output;
}

typedef struct {
    char labels[64];
    int num_labels;
} EinsumSide;

/**
 * Parse one einsum subscript (alphabetic labels only) into `side`. -1 on a bad
 * character or more than 64 labels.
 * @param str (const char*)
 * @param len (int)
 * @param side (EinsumSide*)
 * @return (int)
 */
static int parse_einsum_side(const char* str, int len, EinsumSide* side) {
    side->num_labels = 0;
    for (int i = 0; i < len; i++) {
        if (!isalpha((unsigned char)str[i]))
            return -1;
        if (side->num_labels >= 64)
            return -1;
        side->labels[side->num_labels++] = str[i];
    }
    return 0;
}

/**
 * Evaluate an explicit einsum `equation` ("ij,jk->ik") over the operands by brute
 * force over all label combinations (eager). Returns NULL on a malformed equation,
 * operand/label mismatch, or inconsistent label sizes.
 * @param equation (const char*)
 * @param tensors (Tensor**)
 * @param num_tensors (int)
 * @return (Tensor*)
 */
Tensor* tensor_einsum(const char* equation, Tensor** tensors, int num_tensors) {
    if (!equation || !tensors || num_tensors <= 0)
        return NULL;

    const char* arrow = strstr(equation, "->");
    if (!arrow) {
        LOG_ERROR("tensor_einsum: equation must contain '->'");
        return NULL;
    }

    const char* lhs = equation;
    int lhs_len     = (int)(arrow - equation);
    const char* rhs = arrow + 2;
    int rhs_len     = (int)strlen(rhs);

    const char* input_strs[16];
    int input_lens[16];
    int num_inputs = 0;

    int start = 0;
    for (int i = 0; i <= lhs_len; i++) {
        if (i == lhs_len || lhs[i] == ',') {
            if (num_inputs == 16) {
                LOG_ERROR("tensor_einsum: more than 16 operands");
                return NULL;
            }
            input_strs[num_inputs] = lhs + start;
            input_lens[num_inputs] = i - start;
            num_inputs++;
            start = i + 1;
        }
    }

    if (num_inputs != num_tensors) {
        LOG_ERROR("tensor_einsum: %d operands in equation but %d tensors", num_inputs, num_tensors);
        return NULL;
    }

    EinsumSide inputs[16];
    EinsumSide output;

    for (int i = 0; i < num_inputs; i++) {
        if (parse_einsum_side(input_strs[i], input_lens[i], &inputs[i]) != 0)
            return NULL;
        if (inputs[i].num_labels != tensors[i]->ndim) {
            LOG_ERROR("tensor_einsum: operand %d has %d dims but subscript has %d labels", i,
                      tensors[i]->ndim, inputs[i].num_labels);
            return NULL;
        }
    }

    if (parse_einsum_side(rhs, rhs_len, &output) != 0)
        return NULL;

    char all_labels[128];
    int label_sizes[128];
    int num_unique = 0;

    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < inputs[i].num_labels; j++) {
            char c    = inputs[i].labels[j];
            int size  = tensors[i]->shape[j];
            int found = -1;
            for (int k = 0; k < num_unique; k++) {
                if (all_labels[k] == c) {
                    found = k;
                    break;
                }
            }
            if (found >= 0) {
                if (label_sizes[found] != size) {
                    LOG_ERROR("tensor_einsum: inconsistent size for label '%c'", c);
                    return NULL;
                }
            } else {
                all_labels[num_unique]  = c;
                label_sizes[num_unique] = size;
                num_unique++;
            }
        }
    }

    int out_ndim = output.num_labels;
    int out_shape[64];
    for (int i = 0; i < out_ndim; i++) {
        int found = -1;
        for (int k = 0; k < num_unique; k++) {
            if (all_labels[k] == output.labels[i]) {
                found = k;
                break;
            }
        }
        if (found < 0) {
            LOG_ERROR("tensor_einsum: output label '%c' not in inputs", output.labels[i]);
            return NULL;
        }
        out_shape[i] = label_sizes[found];
    }

    if (out_ndim == 0) {
        out_ndim     = 1;
        out_shape[0] = 1;
    }

    for (int i = 0; i < num_tensors; i++) {
        tensor_ensure_executed(tensors[i]);
        if (!tensors[i]->data)
            return NULL;
    }

    TensorConfig config = {.dtype      = tensors[0]->dtype,
                           .device     = tensors[0]->device,
                           .has_dtype  = true,
                           .has_device = true};
    Tensor* result      = tensor_zeros(out_shape, out_ndim, &config);
    if (!result)
        return NULL;

    float* out_data = (float*)tensor_data_ptr(result);

    int label_indices[128];
    int total_iters = 1;
    for (int k = 0; k < num_unique; k++) {
        total_iters *= label_sizes[k];
        label_indices[k] = 0;
    }

    memset(label_indices, 0, num_unique * sizeof(int));

    for (int iter = 0; iter < total_iters; iter++) {
        float product = 1.0f;
        for (int t = 0; t < num_tensors; t++) {
            size_t flat   = 0;
            size_t stride = 1;
            for (int d = inputs[t].num_labels - 1; d >= 0; d--) {
                char c = inputs[t].labels[d];
                int k;
                for (k = 0; k < num_unique; k++) {
                    if (all_labels[k] == c)
                        break;
                }
                flat += (size_t)label_indices[k] * stride;
                stride *= (size_t)tensors[t]->shape[d];
            }
            product *= tensor_get_float(tensors[t], flat);
        }

        size_t out_flat    = 0;
        size_t out_stride  = 1;
        bool scalar_output = (output.num_labels == 0);
        if (!scalar_output) {
            for (int d = output.num_labels - 1; d >= 0; d--) {
                char c = output.labels[d];
                int k;
                for (k = 0; k < num_unique; k++) {
                    if (all_labels[k] == c)
                        break;
                }
                out_flat += (size_t)label_indices[k] * out_stride;
                out_stride *= (size_t)out_shape[d];
            }
        }

        out_data[out_flat] += product;

        for (int k = num_unique - 1; k >= 0; k--) {
            label_indices[k]++;
            if (label_indices[k] < label_sizes[k])
                break;
            label_indices[k] = 0;
        }
    }

    if (output.num_labels == 0) {
        int scalar_shape[] = {1};
        Tensor* scalar     = tensor_full(scalar_shape, 1, &config, out_data[0]);
        tensor_free(result);
        return scalar;
    }

    return result;
}

/* ── In-place elementwise ops (eager) ───────────────────────────────────────
 * a op= b, mutating a's realized buffer directly (no new IR node / no alloc) -
 * for optimizer/manual updates and gradient accumulation. Realizes both inputs
 * first, then SIMD-updates a. f32 only; b may be a's shape, a scalar [1], or a
 * contiguous trailing broadcast (i % b->numel). Returns a, or NULL on error.
 * These operate on the materialized buffer and do NOT participate in autograd. */
#include "tensor/realize.h"
#include "ops/simd_math.h"

typedef enum { CML_IP_ADD, CML_IP_SUB, CML_IP_MUL, CML_IP_DIV } CMLInplaceKind;

/**
 * Shared implementation of the in-place binary ops (see the block above): realize
 * both operands, then apply `k` to `a`'s f32 buffer. Returns `a`, or NULL on error.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @param k (CMLInplaceKind)
 * @return (Tensor*)
 */
static Tensor* cml_inplace_binary(Tensor* a, Tensor* b, CMLInplaceKind k) {
    if (!a || !b)
        return NULL;
    if (tensor_realize(a) != 0 || tensor_realize(b) != 0)
        return NULL;
    if (!a->data || !b->data)
        return NULL;
    if (a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32) {
        LOG_ERROR("in-place ops are float32-only");
        return NULL;
    }
    float* ad       = (float*)a->data;
    const float* bd = (const float*)b->data;
    size_t an = a->numel, bn = b->numel;
    if (bn == an) {
        switch (k) {
        case CML_IP_ADD:
            simd_add_f32(ad, bd, ad, an);
            break;
        case CML_IP_SUB:
            simd_sub_f32(ad, bd, ad, an);
            break;
        case CML_IP_MUL:
            simd_mul_f32(ad, bd, ad, an);
            break;
        case CML_IP_DIV:
            simd_div_f32(ad, bd, ad, an);
            break;
        }
    } else if (bn == 1) {
        float s = bd[0];
        switch (k) {
        case CML_IP_ADD:
            simd_add_scalar_f32(ad, s, ad, an);
            break;
        case CML_IP_SUB:
            simd_add_scalar_f32(ad, -s, ad, an);
            break;
        case CML_IP_MUL:
            simd_mul_scalar_f32(ad, s, ad, an);
            break;
        case CML_IP_DIV:
            simd_mul_scalar_f32(ad, (s != 0.0f) ? 1.0f / s : 0.0f, ad, an);
            break;
        }
    } else {
        if (bn == 0 || an % bn != 0) {
            LOG_ERROR("in-place: incompatible shapes");
            return NULL;
        }
        for (size_t i = 0; i < an; i++) {
            float bv = bd[i % bn];
            switch (k) {
            case CML_IP_ADD:
                ad[i] += bv;
                break;
            case CML_IP_SUB:
                ad[i] -= bv;
                break;
            case CML_IP_MUL:
                ad[i] *= bv;
                break;
            case CML_IP_DIV:
                ad[i] /= bv;
                break;
            }
        }
    }
    return a;
}

/**
 * In-place `a += b`, mutating `a`'s buffer; returns `a`. Not differentiable.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_add_(Tensor* a, Tensor* b) { return cml_inplace_binary(a, b, CML_IP_ADD); }
/**
 * In-place `a -= b`, mutating `a`'s buffer; returns `a`. Not differentiable.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_sub_(Tensor* a, Tensor* b) { return cml_inplace_binary(a, b, CML_IP_SUB); }
/**
 * In-place `a *= b`, mutating `a`'s buffer; returns `a`. Not differentiable.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_mul_(Tensor* a, Tensor* b) { return cml_inplace_binary(a, b, CML_IP_MUL); }
/**
 * In-place `a /= b`, mutating `a`'s buffer; returns `a`. Not differentiable.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* tensor_div_(Tensor* a, Tensor* b) { return cml_inplace_binary(a, b, CML_IP_DIV); }

/* ── FFT (1-D discrete Fourier transform) ───────────────────────────────────
 * Radix-2 iterative Cooley-Tukey for power-of-two n (O(n log n)); O(n^2) DFT
 * fallback otherwise. In-place on separate real/imag buffers. inverse!=0 does
 * the inverse transform (1/n normalized). */
#include <math.h>

/**
 * In-place 1-D FFT on split real/imag buffers (see the block above): radix-2 for
 * power-of-two `n`, else an O(n^2) DFT. Returns 0 on success, -1 on error.
 * @param re (float*)
 * @param im (float*)
 * @param n (int)
 * @param inverse (int)
 * @return (int)
 */
int cml_fft_1d(float* re, float* im, int n, int inverse) {
    if (n <= 0 || !re || !im)
        return -1;
    if ((n & (n - 1)) == 0) {
        /* bit-reversal permutation */
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j) {
                float t;
                t     = re[i];
                re[i] = re[j];
                re[j] = t;
                t     = im[i];
                im[i] = im[j];
                im[j] = t;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2.0 * M_PI / len * (inverse ? 1.0 : -1.0);
            float wr = (float)cos(ang), wi = (float)sin(ang);
            for (int i = 0; i < n; i += len) {
                float cwr = 1.0f, cwi = 0.0f;
                for (int k = 0; k < len / 2; k++) {
                    float ur = re[i + k], ui = im[i + k];
                    float xr = re[i + k + len / 2], xi = im[i + k + len / 2];
                    float vr            = xr * cwr - xi * cwi;
                    float vi            = xr * cwi + xi * cwr;
                    re[i + k]           = ur + vr;
                    im[i + k]           = ui + vi;
                    re[i + k + len / 2] = ur - vr;
                    im[i + k + len / 2] = ui - vi;
                    float nwr           = cwr * wr - cwi * wi;
                    float nwi           = cwr * wi + cwi * wr;
                    cwr                 = nwr;
                    cwi                 = nwi;
                }
            }
        }
        if (inverse)
            for (int i = 0; i < n; i++) {
                re[i] /= (float)n;
                im[i] /= (float)n;
            }
        return 0;
    }
    /* non-power-of-two: naive DFT into temporaries */
    float* tr = (float*)cml_malloc((size_t)n * sizeof(float));
    float* ti = (float*)cml_malloc((size_t)n * sizeof(float));
    if (!tr || !ti) {
        cml_free(tr);
        cml_free(ti);
        return -1;
    }
    double s = inverse ? 1.0 : -1.0;
    for (int k = 0; k < n; k++) {
        double sr = 0.0, si = 0.0;
        for (int t = 0; t < n; t++) {
            double ang = s * 2.0 * M_PI * (double)k * (double)t / (double)n;
            double c = cos(ang), sn = sin(ang);
            sr += (double)re[t] * c - (double)im[t] * sn;
            si += (double)re[t] * sn + (double)im[t] * c;
        }
        tr[k] = (float)(inverse ? sr / n : sr);
        ti[k] = (float)(inverse ? si / n : si);
    }
    memcpy(re, tr, (size_t)n * sizeof(float));
    memcpy(im, ti, (size_t)n * sizeof(float));
    cml_free(tr);
    cml_free(ti);
    return 0;
}

/**
 * Tensor FFT of a 1-D complex signal stored as [n, 2] (last dim = {real, imag}).
 * Returns a new [n, 2] tensor. inverse!=0 does the inverse transform.
 * @param x (Tensor*)
 * @param inverse (int)
 * @return (Tensor*)
 */
Tensor* cml_fft(Tensor* x, int inverse) {
    if (!x)
        return NULL;
    extern int tensor_realize(Tensor*);
    if (tensor_realize(x) != 0 || !x->data)
        return NULL;
    if (x->ndim < 1 || x->shape[x->ndim - 1] != 2 || x->dtype != DTYPE_FLOAT32) {
        LOG_ERROR("cml_fft: input must be float32 with last dim == 2 (complex)");
        return NULL;
    }
    int n     = (int)(x->numel / 2);
    float* re = (float*)cml_malloc((size_t)n * sizeof(float));
    float* im = (float*)cml_malloc((size_t)n * sizeof(float));
    if (!re || !im) {
        cml_free(re);
        cml_free(im);
        return NULL;
    }
    const float* xd = (const float*)x->data;
    for (int i = 0; i < n; i++) {
        re[i] = xd[2 * i];
        im[i] = xd[2 * i + 1];
    }
    if (cml_fft_1d(re, im, n, inverse) != 0) {
        cml_free(re);
        cml_free(im);
        return NULL;
    }
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = x->device, .has_dtype = true, .has_device = true};
    Tensor* out = tensor_empty(x->shape, x->ndim, &cfg); /* materialized leaf */
    if (out && out->data) {
        float* od = (float*)out->data;
        for (int i = 0; i < n; i++) {
            od[2 * i]     = re[i];
            od[2 * i + 1] = im[i];
        }
        out->is_executed = true; /* holds real data; do not recompute */
    }
    cml_free(re);
    cml_free(im);
    return out;
}

/**
 * 2-D FFT of a complex image stored as [H, W, 2] (last dim = {real, imag}):
 * FFT along W (each row) then along H (each column). Returns a new [H,W,2].
 * @param x (Tensor*)
 * @param inverse (int)
 * @return (Tensor*)
 */
Tensor* cml_fft2(Tensor* x, int inverse) {
    if (!x)
        return NULL;
    extern int tensor_realize(Tensor*);
    if (tensor_realize(x) != 0 || !x->data)
        return NULL;
    if (x->ndim != 3 || x->shape[2] != 2 || x->dtype != DTYPE_FLOAT32) {
        LOG_ERROR("cml_fft2: input must be float32 [H,W,2]");
        return NULL;
    }
    int H = x->shape[0], W = x->shape[1];
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = x->device, .has_dtype = true, .has_device = true};
    Tensor* out = tensor_empty(x->shape, x->ndim, &cfg);
    if (!out || !out->data)
        return out;
    memcpy(out->data, x->data, (size_t)H * W * 2 * sizeof(float));
    float* od = (float*)out->data;
    float* re = (float*)cml_malloc((size_t)(H > W ? H : W) * sizeof(float));
    float* im = (float*)cml_malloc((size_t)(H > W ? H : W) * sizeof(float));
    if (!re || !im) {
        cml_free(re);
        cml_free(im);
        return out;
    }
    /* rows: length-W FFT */
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            re[c] = od[(r * W + c) * 2];
            im[c] = od[(r * W + c) * 2 + 1];
        }
        cml_fft_1d(re, im, W, inverse);
        for (int c = 0; c < W; c++) {
            od[(r * W + c) * 2]     = re[c];
            od[(r * W + c) * 2 + 1] = im[c];
        }
    }
    /* columns: length-H FFT */
    for (int c = 0; c < W; c++) {
        for (int r = 0; r < H; r++) {
            re[r] = od[(r * W + c) * 2];
            im[r] = od[(r * W + c) * 2 + 1];
        }
        cml_fft_1d(re, im, H, inverse);
        for (int r = 0; r < H; r++) {
            od[(r * W + c) * 2]     = re[r];
            od[(r * W + c) * 2 + 1] = im[r];
        }
    }
    cml_free(re);
    cml_free(im);
    out->is_executed = true;
    return out;
}

/**
 * Real-input FFT of a 1-D float signal [n]; returns the n/2+1 non-redundant bins as [n/2+1, 2].
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_rfft(Tensor* x) {
    if (!x)
        return NULL;
    extern int tensor_realize(Tensor*);
    if (tensor_realize(x) != 0 || !x->data)
        return NULL;
    if (x->ndim != 1 || x->dtype != DTYPE_FLOAT32) {
        LOG_ERROR("cml_rfft: input must be a float32 1-D signal");
        return NULL;
    }
    int n     = x->shape[0];
    float* re = (float*)cml_calloc((size_t)n, sizeof(float));
    float* im = (float*)cml_calloc((size_t)n, sizeof(float));
    if (!re || !im) {
        cml_free(re);
        cml_free(im);
        return NULL;
    }
    memcpy(re, x->data, (size_t)n * sizeof(float));
    if (cml_fft_1d(re, im, n, 0) != 0) {
        cml_free(re);
        cml_free(im);
        return NULL;
    }
    int bins         = n / 2 + 1;
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = x->device, .has_dtype = true, .has_device = true};
    int shape[] = {bins, 2};
    Tensor* out = tensor_empty(shape, 2, &cfg);
    if (out && out->data) {
        float* od = (float*)out->data;
        for (int k = 0; k < bins; k++) {
            od[2 * k]     = re[k];
            od[2 * k + 1] = im[k];
        }
        out->is_executed = true;
    }
    cml_free(re);
    cml_free(im);
    return out;
}

/**
 * Inverse real FFT: reconstructs the length-`n` real signal from its n/2+1 complex bins
 * [n/2+1, 2] using Hermitian symmetry. NULL if the bin count does not match n.
 * @param x (Tensor*)
 * @param n (int)
 * @return (Tensor*)
 */
Tensor* cml_irfft(Tensor* x, int n) {
    if (!x)
        return NULL;
    extern int tensor_realize(Tensor*);
    if (tensor_realize(x) != 0 || !x->data)
        return NULL;
    if (x->ndim != 2 || x->shape[1] != 2 || x->dtype != DTYPE_FLOAT32 || x->shape[0] != n / 2 + 1) {
        LOG_ERROR("cml_irfft: input must be float32 [n/2+1, 2] matching n");
        return NULL;
    }
    float* re = (float*)cml_calloc((size_t)n, sizeof(float));
    float* im = (float*)cml_calloc((size_t)n, sizeof(float));
    if (!re || !im) {
        cml_free(re);
        cml_free(im);
        return NULL;
    }
    const float* xd = (const float*)x->data;
    int bins        = n / 2 + 1;
    for (int k = 0; k < bins; k++) {
        re[k] = xd[2 * k];
        im[k] = xd[2 * k + 1];
    }
    for (int k = bins; k < n; k++) { /* Hermitian: X[n-k] = conj(X[k]) */
        re[k] = re[n - k];
        im[k] = -im[n - k];
    }
    if (cml_fft_1d(re, im, n, 1) != 0) {
        cml_free(re);
        cml_free(im);
        return NULL;
    }
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = x->device, .has_dtype = true, .has_device = true};
    int shape[] = {n};
    Tensor* out = tensor_empty(shape, 1, &cfg);
    if (out && out->data) {
        float* od = (float*)out->data;
        for (int i = 0; i < n; i++)
            od[i] = re[i];
        out->is_executed = true;
    }
    cml_free(re);
    cml_free(im);
    return out;
}

/** Full linear convolution (length na+nb-1) of two real signals via FFT; NULL on error. */
static float* fft_convolve_full(const float* a, int na, const float* b, int nb, int* out_len) {
    int L      = na + nb - 1;
    *out_len   = L;
    float* ar  = (float*)cml_calloc((size_t)L, sizeof(float));
    float* ai  = (float*)cml_calloc((size_t)L, sizeof(float));
    float* br  = (float*)cml_calloc((size_t)L, sizeof(float));
    float* bi  = (float*)cml_calloc((size_t)L, sizeof(float));
    float* out = (float*)cml_malloc((size_t)L * sizeof(float));
    if (!ar || !ai || !br || !bi || !out) {
        cml_free(ar);
        cml_free(ai);
        cml_free(br);
        cml_free(bi);
        cml_free(out);
        return NULL;
    }
    memcpy(ar, a, (size_t)na * sizeof(float));
    memcpy(br, b, (size_t)nb * sizeof(float));
    if (cml_fft_1d(ar, ai, L, 0) == 0 && cml_fft_1d(br, bi, L, 0) == 0) {
        for (int i = 0; i < L; i++) {
            float cr = ar[i] * br[i] - ai[i] * bi[i];
            float ci = ar[i] * bi[i] + ai[i] * br[i];
            ar[i]    = cr;
            ai[i]    = ci;
        }
        cml_fft_1d(ar, ai, L, 1);
        for (int i = 0; i < L; i++)
            out[i] = ar[i];
    } else {
        cml_free(out);
        out = NULL;
    }
    cml_free(ar);
    cml_free(ai);
    cml_free(br);
    cml_free(bi);
    return out;
}

/** Build the output tensor for a given convolution `mode` (0 full, 1 same, 2 valid). */
static Tensor* conv_trim(const float* full, int na, int nb, int mode, int device) {
    int L = na + nb - 1;
    int out_len, start;
    if (mode == 0) {
        out_len = L;
        start   = 0;
    } else if (mode == 1) {
        out_len = na > nb ? na : nb;
        start   = (L - out_len) / 2;
    } else {
        int big = na > nb ? na : nb, small = na > nb ? nb : na;
        out_len = big - small + 1;
        start   = small - 1;
    }
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = device, .has_dtype = true, .has_device = true};
    int shape[] = {out_len};
    Tensor* out = tensor_empty(shape, 1, &cfg);
    if (out && out->data) {
        float* od = (float*)out->data;
        for (int i = 0; i < out_len; i++)
            od[i] = full[start + i];
        out->is_executed = true;
    }
    return out;
}

/** Validate a pair of 1-D float signals and return their flat data, else 0. */
static int conv_inputs(Tensor* a, Tensor* b, const float** ad, int* na, const float** bd, int* nb) {
    extern int tensor_realize(Tensor*);
    if (!a || !b || tensor_realize(a) != 0 || tensor_realize(b) != 0 || !a->data || !b->data)
        return 0;
    if (a->ndim != 1 || b->ndim != 1 || a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32)
        return 0;
    *ad = (const float*)a->data;
    *bd = (const float*)b->data;
    *na = a->shape[0];
    *nb = b->shape[0];
    return 1;
}

/**
 * 1-D convolution of two real signals (NumPy convention, kernel flipped). mode: 0 full,
 * 1 same (length max of inputs), 2 valid. NULL on error.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @param mode (int)
 * @return (Tensor*)
 */
Tensor* cml_convolve1d(Tensor* a, Tensor* b, int mode) {
    const float *ad, *bd;
    int na, nb;
    if (!conv_inputs(a, b, &ad, &na, &bd, &nb)) {
        LOG_ERROR("cml_convolve1d: inputs must be float32 1-D signals");
        return NULL;
    }
    int L;
    float* full = fft_convolve_full(ad, na, bd, nb, &L);
    if (!full)
        return NULL;
    Tensor* out = conv_trim(full, na, nb, mode, a->device);
    cml_free(full);
    return out;
}

/**
 * 1-D cross-correlation of two real signals (NumPy convention, kernel not flipped). mode:
 * 0 full, 1 same, 2 valid. NULL on error.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @param mode (int)
 * @return (Tensor*)
 */
Tensor* cml_correlate1d(Tensor* a, Tensor* b, int mode) {
    const float *ad, *bd;
    int na, nb;
    if (!conv_inputs(a, b, &ad, &na, &bd, &nb)) {
        LOG_ERROR("cml_correlate1d: inputs must be float32 1-D signals");
        return NULL;
    }
    float* rev =
        (float*)cml_malloc((size_t)nb * sizeof(float)); /* correlate = convolve with b reversed */
    if (!rev)
        return NULL;
    for (int i = 0; i < nb; i++)
        rev[i] = bd[nb - 1 - i];
    int L;
    float* full = fft_convolve_full(ad, na, rev, nb, &L);
    cml_free(rev);
    if (!full)
        return NULL;
    Tensor* out = conv_trim(full, na, nb, mode, a->device);
    cml_free(full);
    return out;
}

/**
 * Histogram of all elements into `bins` equal-width bins over [lo,hi] (values outside are
 * dropped; hi falls in the last bin). If lo>=hi the range is taken from the data. Counts [bins].
 * @param a (Tensor*)
 * @param bins (int)
 * @param lo (float)
 * @param hi (float)
 * @return (Tensor*)
 */
Tensor* cml_histogram(Tensor* a, int bins, float lo, float hi) {
    if (!a || bins <= 0) {
        LOG_ERROR("cml_histogram: need a tensor and bins>0");
        return NULL;
    }
    extern int tensor_realize(Tensor*);
    if (tensor_realize(a) != 0 || !a->data)
        return NULL;
    if (lo >= hi) {
        lo = INFINITY;
        hi = -INFINITY;
        for (size_t i = 0; i < a->numel; i++) {
            float v = tensor_get_float(a, i);
            if (v < lo)
                lo = v;
            if (v > hi)
                hi = v;
        }
        if (lo >= hi)
            hi = lo + 1.0f;
    }
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = a->device, .has_dtype = true, .has_device = true};
    int shape[] = {bins};
    Tensor* out = tensor_empty(shape, 1, &cfg);
    if (!out || !out->data)
        return out;
    float* counts = (float*)out->data;
    memset(counts, 0, (size_t)bins * sizeof(float));
    float width = (hi - lo) / bins;
    for (size_t i = 0; i < a->numel; i++) {
        float v = tensor_get_float(a, i);
        if (v < lo || v > hi)
            continue;
        int idx = (int)((v - lo) / width);
        if (idx >= bins)
            idx = bins - 1;
        counts[idx] += 1.0f;
    }
    out->is_executed = true;
    return out;
}

/**
 * Count occurrences of each non-negative integer value; output length is max(max_value+1,
 * minlength). Counts [length].
 * @param a (Tensor*)
 * @param minlength (int)
 * @return (Tensor*)
 */
Tensor* cml_bincount(Tensor* a, int minlength) {
    if (!a) {
        LOG_ERROR("cml_bincount: need a tensor");
        return NULL;
    }
    extern int tensor_realize(Tensor*);
    if (tensor_realize(a) != 0 || !a->data)
        return NULL;
    int maxv = -1;
    for (size_t i = 0; i < a->numel; i++) {
        int v = (int)(tensor_get_float(a, i) + 0.5f);
        if (v < 0) {
            LOG_ERROR("cml_bincount: values must be non-negative");
            return NULL;
        }
        if (v > maxv)
            maxv = v;
    }
    int length = maxv + 1;
    if (length < minlength)
        length = minlength;
    if (length <= 0)
        length = minlength > 0 ? minlength : 0;
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = a->device, .has_dtype = true, .has_device = true};
    int shape[] = {length};
    Tensor* out = tensor_empty(shape, 1, &cfg);
    if (!out || !out->data || length == 0)
        return out;
    float* counts = (float*)out->data;
    memset(counts, 0, (size_t)length * sizeof(float));
    for (size_t i = 0; i < a->numel; i++)
        counts[(int)(tensor_get_float(a, i) + 0.5f)] += 1.0f;
    out->is_executed = true;
    return out;
}

/** Insertion index of `value` into ascending `sorted[0..n)`; right!=0 inserts after equals. */
static int searchsorted_buf(const float* sorted, int n, float value, int right) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid      = lo + (hi - lo) / 2;
        int go_right = right ? (sorted[mid] <= value) : (sorted[mid] < value);
        if (go_right)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

/**
 * Index at which `value` would keep ascending `sorted` ordered; right!=0 inserts after equals.
 * @param sorted (Tensor*)
 * @param value (float)
 * @param right (int)
 * @return (int)
 */
int cml_searchsorted(Tensor* sorted, float value, int right) {
    if (!sorted || sorted->ndim != 1) {
        LOG_ERROR("cml_searchsorted: input must be a 1-D sorted tensor");
        return 0;
    }
    extern int tensor_realize(Tensor*);
    if (tensor_realize(sorted) != 0 || !sorted->data)
        return 0;
    int n      = sorted->shape[0];
    float* buf = (float*)cml_malloc((size_t)n * sizeof(float));
    if (!buf)
        return 0;
    for (int i = 0; i < n; i++)
        buf[i] = tensor_get_float(sorted, i);
    int idx = searchsorted_buf(buf, n, value, right);
    cml_free(buf);
    return idx;
}

/**
 * Piecewise-linear interpolation of query points `x` against sample points `xp` (ascending)
 * with values `fp`; queries outside xp are clamped to the endpoints. Returns values [len(x)].
 * @param x (Tensor*)
 * @param xp (Tensor*)
 * @param fp (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_interp1d(Tensor* x, Tensor* xp, Tensor* fp) {
    if (!x || !xp || !fp || x->ndim != 1 || xp->ndim != 1 || fp->ndim != 1 ||
        xp->shape[0] != fp->shape[0] || xp->shape[0] < 1) {
        LOG_ERROR("cml_interp1d: need 1-D x, and xp,fp of equal length");
        return NULL;
    }
    extern int tensor_realize(Tensor*);
    if (tensor_realize(x) != 0 || tensor_realize(xp) != 0 || tensor_realize(fp) != 0)
        return NULL;
    int nq = x->shape[0], np = xp->shape[0];
    float* xpb = (float*)cml_malloc((size_t)np * sizeof(float));
    float* fpb = (float*)cml_malloc((size_t)np * sizeof(float));
    if (!xpb || !fpb) {
        cml_free(xpb);
        cml_free(fpb);
        return NULL;
    }
    for (int i = 0; i < np; i++) {
        xpb[i] = tensor_get_float(xp, i);
        fpb[i] = tensor_get_float(fp, i);
    }
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = x->device, .has_dtype = true, .has_device = true};
    int shape[] = {nq};
    Tensor* out = tensor_empty(shape, 1, &cfg);
    if (out && out->data) {
        float* od = (float*)out->data;
        for (int i = 0; i < nq; i++) {
            float q = tensor_get_float(x, i);
            if (q <= xpb[0]) {
                od[i] = fpb[0];
            } else if (q >= xpb[np - 1]) {
                od[i] = fpb[np - 1];
            } else {
                int j = searchsorted_buf(xpb, np, q, 1); /* first sample strictly greater */
                if (j < 1)
                    j = 1;
                float t = (q - xpb[j - 1]) / (xpb[j] - xpb[j - 1]);
                od[i]   = fpb[j - 1] + t * (fpb[j] - fpb[j - 1]);
            }
        }
        out->is_executed = true;
    }
    cml_free(xpb);
    cml_free(fpb);
    return out;
}

/** Allocate a 1-D window tensor of length N, defaulting to float32/CPU when config is NULL. */
static Tensor* window_tensor(int N, const TensorConfig* config) {
    TensorConfig def = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int shape[] = {N};
    return tensor_empty(shape, 1, config ? config : &def);
}

/** Fill a generalized cosine window w[i] = a0 - a1*cos(2pi i/(N-1)) + a2*cos(4pi i/(N-1)). */
static void fill_cosine_window(float* w, int N, float a0, float a1, float a2) {
    if (N == 1) {
        w[0] = 1.0f;
        return;
    }
    for (int i = 0; i < N; i++) {
        float x = 2.0f * (float)M_PI * i / (N - 1);
        w[i]    = a0 - a1 * cosf(x) + a2 * cosf(2.0f * x);
    }
}

/**
 * Hann window of length N (NumPy np.hanning convention). NULL if N<1.
 * @param N (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_hann_window(int N, const TensorConfig* config) {
    if (N < 1)
        return NULL;
    Tensor* out = window_tensor(N, config);
    if (out && out->data) {
        fill_cosine_window((float*)out->data, N, 0.5f, 0.5f, 0.0f);
        out->is_executed = true;
    }
    return out;
}

/**
 * Hamming window of length N (NumPy np.hamming convention). NULL if N<1.
 * @param N (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_hamming_window(int N, const TensorConfig* config) {
    if (N < 1)
        return NULL;
    Tensor* out = window_tensor(N, config);
    if (out && out->data) {
        fill_cosine_window((float*)out->data, N, 0.54f, 0.46f, 0.0f);
        out->is_executed = true;
    }
    return out;
}

/**
 * Blackman window of length N (NumPy np.blackman convention). NULL if N<1.
 * @param N (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_blackman_window(int N, const TensorConfig* config) {
    if (N < 1)
        return NULL;
    Tensor* out = window_tensor(N, config);
    if (out && out->data) {
        fill_cosine_window((float*)out->data, N, 0.42f, 0.5f, 0.08f);
        out->is_executed = true;
    }
    return out;
}

/**
 * Bartlett (triangular) window of length N (NumPy np.bartlett convention). NULL if N<1.
 * @param N (int)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_bartlett_window(int N, const TensorConfig* config) {
    if (N < 1)
        return NULL;
    Tensor* out = window_tensor(N, config);
    if (out && out->data) {
        float* w = (float*)out->data;
        if (N == 1) {
            w[0] = 1.0f;
        } else {
            float half = (N - 1) / 2.0f;
            for (int i = 0; i < N; i++)
                w[i] = 1.0f - fabsf((i - half) / half);
        }
        out->is_executed = true;
    }
    return out;
}

/**
 * DFT sample frequencies for a length-n FFT with sample spacing d (NumPy np.fft.fftfreq).
 * @param n (int)
 * @param d (float)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_fftfreq(int n, float d, const TensorConfig* config) {
    if (n < 1 || d == 0.0f)
        return NULL;
    Tensor* out = window_tensor(n, config);
    if (out && out->data) {
        float* f   = (float*)out->data;
        float norm = 1.0f / (d * n);
        int half   = (n - 1) / 2 + 1; /* count of non-negative frequencies */
        for (int i = 0; i < n; i++)
            f[i] = (i < half ? i : i - n) * norm;
        out->is_executed = true;
    }
    return out;
}

/**
 * Non-negative DFT frequencies for a length-n real FFT with spacing d (NumPy np.fft.rfftfreq).
 * @param n (int)
 * @param d (float)
 * @param config (const TensorConfig*)
 * @return (Tensor*)
 */
Tensor* cml_rfftfreq(int n, float d, const TensorConfig* config) {
    if (n < 1 || d == 0.0f)
        return NULL;
    int bins    = n / 2 + 1;
    Tensor* out = window_tensor(bins, config);
    if (out && out->data) {
        float* f   = (float*)out->data;
        float norm = 1.0f / (d * n);
        for (int i = 0; i < bins; i++)
            f[i] = i * norm;
        out->is_executed = true;
    }
    return out;
}

/**
 * Analytic signal of a real 1-D signal via the Hilbert transform, returned as [n,2] complex.
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_hilbert(Tensor* x) {
    if (!x)
        return NULL;
    extern int tensor_realize(Tensor*);
    if (tensor_realize(x) != 0 || !x->data || x->ndim != 1 || x->dtype != DTYPE_FLOAT32) {
        LOG_ERROR("cml_hilbert: input must be a float32 1-D signal");
        return NULL;
    }
    int n            = x->shape[0];
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = x->device, .has_dtype = true, .has_device = true};
    int cshape[] = {n, 2};
    Tensor* cin  = tensor_empty(cshape, 2, &cfg);
    if (!cin || !cin->data) {
        tensor_free(cin);
        return NULL;
    }
    float* cd = (float*)cin->data;
    for (int i = 0; i < n; i++) {
        cd[2 * i]     = (float)tensor_get_float(x, i);
        cd[2 * i + 1] = 0.0f;
    }
    cin->is_executed = true;
    Tensor* F        = cml_fft(cin, 0);
    tensor_free(cin);
    if (!F)
        return NULL;
    float* fd = (float*)tensor_data_ptr(F);
    for (int k = 0; k < n; k++) { /* double positive freqs, zero negative (scipy.signal.hilbert) */
        float m;
        if (k == 0 || (n % 2 == 0 && k == n / 2))
            m = 1.0f;
        else if (k < (n + 1) / 2)
            m = 2.0f;
        else
            m = 0.0f;
        fd[2 * k] *= m;
        fd[2 * k + 1] *= m;
    }
    Tensor* out = cml_fft(F, 1);
    tensor_free(F);
    return out;
}

/**
 * One-sided power spectral density (periodogram) of a real 1-D signal: mean-detrended, density
 * scaled by 1/(fs*n) with the standard one-sided doubling. Returns [n/2+1]. NULL on error.
 * @param x (Tensor*)
 * @param fs (float)
 * @return (Tensor*)
 */
Tensor* cml_periodogram(Tensor* x, float fs) {
    if (!x)
        return NULL;
    extern int tensor_realize(Tensor*);
    if (tensor_realize(x) != 0 || !x->data || x->ndim != 1 || x->dtype != DTYPE_FLOAT32 ||
        fs <= 0.0f) {
        LOG_ERROR("cml_periodogram: need a float32 1-D signal and fs>0");
        return NULL;
    }
    int n     = x->shape[0];
    double mu = 0.0;
    for (int i = 0; i < n; i++)
        mu += tensor_get_float(x, i);
    mu /= n;
    int dshape[]     = {n};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = x->device, .has_dtype = true, .has_device = true};
    Tensor* xm = tensor_empty(dshape, 1, &cfg);
    if (!xm || !xm->data)
        return NULL;
    float* xmd = (float*)xm->data;
    for (int i = 0; i < n; i++)
        xmd[i] = (float)(tensor_get_float(x, i) - mu); /* detrend: constant */
    xm->is_executed = true;
    Tensor* R       = cml_rfft(xm);
    tensor_free(xm);
    if (!R)
        return NULL;
    int bins     = n / 2 + 1;
    float* rd    = (float*)tensor_data_ptr(R);
    int pshape[] = {bins};
    Tensor* pxx  = tensor_empty(pshape, 1, &cfg);
    if (pxx && pxx->data) {
        float* p    = (float*)pxx->data;
        float scale = 1.0f / (fs * n);
        for (int k = 0; k < bins; k++) {
            float mag2 = rd[2 * k] * rd[2 * k] + rd[2 * k + 1] * rd[2 * k + 1];
            float v    = scale * mag2;
            if (k != 0 && !(n % 2 == 0 && k == n / 2))
                v *= 2.0f; /* one-sided: double all but DC and Nyquist */
            p[k] = v;
        }
        pxx->is_executed = true;
    }
    tensor_free(R);
    return pxx;
}

/**
 * Resample a real 1-D signal to `num` samples by Fourier-domain truncation/zero-padding. NULL on
 * error.
 * @param x (Tensor*)
 * @param num (int)
 * @return (Tensor*)
 */
Tensor* cml_resample(Tensor* x, int num) {
    if (!x)
        return NULL;
    extern int tensor_realize(Tensor*);
    if (tensor_realize(x) != 0 || !x->data || x->ndim != 1 || x->dtype != DTYPE_FLOAT32 ||
        num < 1) {
        LOG_ERROR("cml_resample: need a float32 1-D signal and num>=1");
        return NULL;
    }
    int n     = x->shape[0];
    Tensor* R = cml_rfft(x);
    if (!R)
        return NULL;
    int in_bins      = n / 2 + 1;
    int out_bins     = num / 2 + 1;
    float* rd        = (float*)tensor_data_ptr(R);
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = x->device, .has_dtype = true, .has_device = true};
    int yshape[] = {out_bins, 2};
    Tensor* Y    = tensor_empty(yshape, 2, &cfg);
    if (!Y || !Y->data) {
        tensor_free(R);
        tensor_free(Y);
        return NULL;
    }
    float* yd = (float*)Y->data;
    for (int i = 0; i < out_bins * 2; i++)
        yd[i] = 0.0f;
    int N   = num < n ? num : n;
    int nyq = N / 2 + 1;
    for (int k = 0; k < nyq && k < in_bins && k < out_bins; k++) {
        yd[2 * k]     = rd[2 * k];
        yd[2 * k + 1] = rd[2 * k + 1];
    }
    if (N % 2 == 0) { /* split the shared Nyquist bin so energy is not doubled */
        int kn = N / 2;
        if (kn < out_bins) {
            yd[2 * kn] *= 0.5f;
            yd[2 * kn + 1] *= 0.5f;
        }
    }
    Y->is_executed = true;
    tensor_free(R);
    Tensor* z = cml_irfft(Y, num);
    tensor_free(Y);
    if (!z)
        return NULL;
    float* zd   = (float*)tensor_data_ptr(z);
    float scale = (float)num / n;
    for (int i = 0; i < num; i++)
        zd[i] *= scale;
    return z;
}

/** Ascending float comparator for qsort (median-filter window sort). */
static int float_asc_cmp_sig(const void* x, const void* y) {
    float a = *(const float*)x, b = *(const float*)y;
    return (a > b) - (a < b);
}

/** Build a 1-D float tensor from a buffer with float32/CPU-or-device config. */
static Tensor* sig_vec(const float* data, int n, int device) {
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = device, .has_dtype = true, .has_device = true};
    int shape[] = {n};
    return tensor_from_data(data, shape, 1, &cfg);
}

/**
 * Apply an IIR/FIR difference-equation filter with numerator b and denominator a to signal x.
 * @param b (Tensor*)
 * @param a (Tensor*)
 * @param x (Tensor*)
 * @return (Tensor*)
 */
Tensor* cml_lfilter(Tensor* b, Tensor* a, Tensor* x) {
    if (!b || !a || !x || b->ndim != 1 || a->ndim != 1 || x->ndim != 1 || a->shape[0] < 1) {
        LOG_ERROR("cml_lfilter: b, a, x must be 1-D and a non-empty");
        return NULL;
    }
    tensor_ensure_executed(b);
    tensor_ensure_executed(a);
    tensor_ensure_executed(x);
    if (!b->data || !a->data || !x->data)
        return NULL;
    int nb = b->shape[0], na = a->shape[0], n = x->shape[0];
    float a0 = tensor_get_float(a, 0);
    if (a0 == 0.0f) {
        LOG_ERROR("cml_lfilter: a[0] must be non-zero");
        return NULL;
    }
    float* y = cml_malloc((size_t)n * sizeof(float));
    if (!y)
        return NULL;
    for (int i = 0; i < n; i++) {
        double acc = 0.0;
        for (int j = 0; j < nb; j++)
            if (i - j >= 0)
                acc += (double)tensor_get_float(b, j) * tensor_get_float(x, i - j);
        for (int j = 1; j < na; j++)
            if (i - j >= 0)
                acc -= (double)tensor_get_float(a, j) * y[i - j];
        y[i] = (float)(acc / a0);
    }
    Tensor* out = sig_vec(y, n, x->device);
    cml_free(y);
    return out;
}

/**
 * Median filter of a 1-D signal with an odd kernel size (zero-padded edges, SciPy convention).
 * @param x (Tensor*)
 * @param kernel (int)
 * @return (Tensor*)
 */
Tensor* cml_medfilt(Tensor* x, int kernel) {
    if (!x || x->ndim != 1 || kernel < 1 || kernel % 2 == 0) {
        LOG_ERROR("cml_medfilt: need a 1-D signal and an odd kernel size");
        return NULL;
    }
    tensor_ensure_executed(x);
    if (!x->data)
        return NULL;
    int n    = x->shape[0];
    int half = kernel / 2;
    float* y = cml_malloc((size_t)n * sizeof(float));
    float* w = cml_malloc((size_t)kernel * sizeof(float));
    if (!y || !w) {
        cml_free(y);
        cml_free(w);
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < kernel; j++) {
            int idx = i - half + j;
            w[j]    = (idx >= 0 && idx < n) ? tensor_get_float(x, idx) : 0.0f; /* zero pad */
        }
        qsort(w, (size_t)kernel, sizeof(float), float_asc_cmp_sig);
        y[i] = w[half];
    }
    Tensor* out = sig_vec(y, n, x->device);
    cml_free(y);
    cml_free(w);
    return out;
}

/**
 * Savitzky-Golay smoothing filter via a local polynomial fit per point (SciPy mode='interp').
 * @param x (Tensor*)
 * @param window_length (int)
 * @param polyorder (int)
 * @return (Tensor*)
 */
Tensor* cml_savgol_filter(Tensor* x, int window_length, int polyorder) {
    if (!x || x->ndim != 1 || window_length < 1 || window_length % 2 == 0 ||
        polyorder >= window_length || polyorder < 0 || window_length > x->shape[0]) {
        LOG_ERROR("cml_savgol_filter: need odd window_length, polyorder<window_length<=n");
        return NULL;
    }
    tensor_ensure_executed(x);
    if (!x->data)
        return NULL;
    int n     = x->shape[0];
    int half  = window_length / 2;
    float* xs = cml_malloc((size_t)window_length * sizeof(float));
    float* ys = cml_malloc((size_t)window_length * sizeof(float));
    float* y  = cml_malloc((size_t)n * sizeof(float));
    if (!xs || !ys || !y) {
        cml_free(xs);
        cml_free(ys);
        cml_free(y);
        return NULL;
    }
    for (int i = 0; i < window_length; i++)
        xs[i] = (float)i;
    Tensor* xt = sig_vec(xs, window_length, x->device);
    for (int i = 0; i < n; i++) {
        int start = i - half;
        if (start < 0)
            start = 0;
        if (start > n - window_length)
            start = n - window_length; /* clamp window to the signal (edge interpolation) */
        for (int j = 0; j < window_length; j++)
            ys[j] = tensor_get_float(x, start + j);
        Tensor* yt     = sig_vec(ys, window_length, x->device);
        Tensor* coeffs = (xt && yt) ? tensor_polyfit(xt, yt, polyorder) : NULL;
        float pos      = (float)(i - start);
        if (coeffs) {
            Tensor* pt  = sig_vec(&pos, 1, x->device);
            Tensor* val = pt ? tensor_polyval(coeffs, pt) : NULL;
            y[i]        = val ? tensor_get_float(val, 0) : 0.0f;
            tensor_free(pt);
            tensor_free(val);
        } else {
            y[i] = ys[i - start];
        }
        tensor_free(yt);
        tensor_free(coeffs);
    }
    tensor_free(xt);
    Tensor* out = sig_vec(y, n, x->device);
    cml_free(xs);
    cml_free(ys);
    cml_free(y);
    return out;
}

/**
 * Zero-crossing rate of a 1-D signal (fraction of adjacent samples whose sign differs).
 * @param x (Tensor*)
 * @return (float)
 */
float cml_zero_crossing_rate(Tensor* x) {
    if (!x || x->ndim != 1 || x->shape[0] < 2)
        return 0.0f;
    tensor_ensure_executed(x);
    if (!x->data)
        return 0.0f;
    int n       = x->shape[0];
    int crosses = 0;
    for (int i = 1; i < n; i++) {
        float a = tensor_get_float(x, i - 1), b = tensor_get_float(x, i);
        if ((a >= 0.0f) != (b >= 0.0f))
            crosses++;
    }
    return (float)crosses / (n - 1);
}

/**
 * Root-mean-square energy sqrt(mean(x^2)) of a 1-D signal.
 * @param x (Tensor*)
 * @return (float)
 */
float cml_rms_energy(Tensor* x) {
    if (!x || x->ndim != 1 || x->shape[0] < 1)
        return 0.0f;
    tensor_ensure_executed(x);
    if (!x->data)
        return 0.0f;
    int n    = x->shape[0];
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        float v = tensor_get_float(x, i);
        s += (double)v * v;
    }
    return (float)sqrt(s / n);
}

/**
 * Spectral centroid (magnitude-weighted mean frequency) of a 1-D signal at sample rate fs.
 * @param x (Tensor*)
 * @param fs (float)
 * @return (float)
 */
float cml_spectral_centroid(Tensor* x, float fs) {
    if (!x || x->ndim != 1 || fs <= 0.0f)
        return 0.0f;
    Tensor* R = cml_rfft(x);
    if (!R)
        return 0.0f;
    int n      = x->shape[0];
    int bins   = n / 2 + 1;
    float* rd  = (float*)tensor_data_ptr(R);
    double num = 0.0, den = 0.0;
    for (int k = 0; k < bins; k++) {
        double mag  = sqrt((double)rd[2 * k] * rd[2 * k] + (double)rd[2 * k + 1] * rd[2 * k + 1]);
        double freq = (double)k * fs / n;
        num += freq * mag;
        den += mag;
    }
    tensor_free(R);
    return den > 0.0 ? (float)(num / den) : 0.0f;
}
