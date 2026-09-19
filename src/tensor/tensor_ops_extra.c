#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "alloc/cml_allocator.h"

Tensor* tensor_where(Tensor* condition, Tensor* x, Tensor* y) {
    if (!condition || !x || !y)
        return NULL;

    WhereParams params = {.cond = condition, .a = x, .b = y};
    return uop_where(&params);
}

Tensor* tensor_one_hot(Tensor* indices, int num_classes) {
    if (!indices)
        return NULL;
    return uop_one_hot(indices, num_classes);
}

Tensor* tensor_roll(Tensor* t, int shift, int axis) {
    if (!t)
        return NULL;
    return uop_roll(t, shift, axis);
}

Tensor* tensor_nonzero(Tensor* t) {
    if (!t)
        return NULL;
    return uop_nonzero(t);
}

Tensor* tensor_copysign(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_copysign(a, b);
}

Tensor* tensor_logaddexp(Tensor* a, Tensor* b) {
    if (!a || !b)
        return NULL;
    return uop_logaddexp(a, b);
}

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
 * a op= b, mutating a's realized buffer directly (no new IR node / no alloc) —
 * for optimizer/manual updates and gradient accumulation. Realizes both inputs
 * first, then SIMD-updates a. f32 only; b may be a's shape, a scalar [1], or a
 * contiguous trailing broadcast (i % b->numel). Returns a, or NULL on error.
 * These operate on the materialized buffer and do NOT participate in autograd. */
#include "tensor/realize.h"
#include "ops/simd_math.h"

typedef enum { CML_IP_ADD, CML_IP_SUB, CML_IP_MUL, CML_IP_DIV } CMLInplaceKind;

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

Tensor* tensor_add_(Tensor* a, Tensor* b) { return cml_inplace_binary(a, b, CML_IP_ADD); }
Tensor* tensor_sub_(Tensor* a, Tensor* b) { return cml_inplace_binary(a, b, CML_IP_SUB); }
Tensor* tensor_mul_(Tensor* a, Tensor* b) { return cml_inplace_binary(a, b, CML_IP_MUL); }
Tensor* tensor_div_(Tensor* a, Tensor* b) { return cml_inplace_binary(a, b, CML_IP_DIV); }

/* ── FFT (1-D discrete Fourier transform) ───────────────────────────────────
 * Radix-2 iterative Cooley-Tukey for power-of-two n (O(n log n)); O(n^2) DFT
 * fallback otherwise. In-place on separate real/imag buffers. inverse!=0 does
 * the inverse transform (1/n normalized). */
#include <math.h>

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

/* Tensor FFT of a 1-D complex signal stored as [n, 2] (last dim = {real, imag}).
 * Returns a new [n, 2] tensor. inverse!=0 does the inverse transform. */
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

/* 2-D FFT of a complex image stored as [H, W, 2] (last dim = {real, imag}):
 * FFT along W (each row) then along H (each column). Returns a new [H,W,2]. */
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
