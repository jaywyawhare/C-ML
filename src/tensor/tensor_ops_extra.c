#include "tensor/tensor.h"
#include "ops/uops.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

Tensor* tensor_where(Tensor* condition, Tensor* x, Tensor* y) {
    if (!condition || !x || !y) return NULL;

    WhereParams params = {.cond = condition, .a = x, .b = y};
    return uop_where(&params);
}

Tensor* tensor_one_hot(Tensor* indices, int num_classes) {
    if (!indices) return NULL;
    return uop_one_hot(indices, num_classes);
}

Tensor* tensor_roll(Tensor* t, int shift, int axis) {
    if (!t) return NULL;
    return uop_roll(t, shift, axis);
}

Tensor* tensor_nonzero(Tensor* t) {
    if (!t) return NULL;
    return uop_nonzero(t);
}

Tensor* tensor_copysign(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    return uop_copysign(a, b);
}

Tensor* tensor_logaddexp(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    return uop_logaddexp(a, b);
}

Tensor* tensor_multinomial(Tensor* probs, int num_samples, bool replacement) {
    if (!probs) return NULL;
    if (probs->ndim != 1 && probs->ndim != 2) {
        LOG_ERROR("tensor_multinomial: probs must be 1D or 2D");
        return NULL;
    }

    tensor_ensure_executed(probs);
    if (!probs->data) return NULL;

    bool batched = (probs->ndim == 2);
    int batch_size = batched ? probs->shape[0] : 1;
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
        out_ndim = 2;
    } else {
        out_shape[0] = num_samples;
        out_ndim = 1;
    }

    TensorConfig config = {
        .dtype = DTYPE_INT32, .device = probs->device,
        .has_dtype = true, .has_device = true
    };
    Tensor* output = tensor_empty(out_shape, out_ndim, &config);
    if (!output) return NULL;

    float* cumsum = (float*)malloc(num_categories * sizeof(float));
    if (!cumsum) { tensor_free(output); return NULL; }

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
            free(cumsum);
            tensor_free(output);
            return NULL;
        }

        for (int c = 0; c < num_categories; c++)
            cumsum[c] /= total;

        bool* used = NULL;
        if (!replacement) {
            used = (bool*)calloc(num_categories, sizeof(bool));
            if (!used) { free(cumsum); tensor_free(output); return NULL; }
        }

        for (int s = 0; s < num_samples; s++) {
            float u = (float)rand() / (float)RAND_MAX;
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
                float acc = 0.0f;
                for (int c = 0; c < num_categories; c++) {
                    if (used[c]) continue;
                    size_t idx = batched ? (size_t)b * num_categories + c : (size_t)c;
                    acc += tensor_get_float(probs, idx);
                    if (acc >= target) { selected = c; break; }
                }
                used[selected] = true;
            } else {
                for (int c = 0; c < num_categories; c++) {
                    if (u <= cumsum[c]) { selected = c; break; }
                }
            }

            size_t out_idx = batched ? (size_t)b * num_samples + s : (size_t)s;
            out_data[out_idx] = selected;
        }

        free(used);
    }

    free(cumsum);
    return output;
}

typedef struct {
    char labels[64];
    int num_labels;
} EinsumSide;

static int parse_einsum_side(const char* str, int len, EinsumSide* side) {
    side->num_labels = 0;
    for (int i = 0; i < len; i++) {
        if (!isalpha((unsigned char)str[i])) return -1;
        if (side->num_labels >= 64) return -1;
        side->labels[side->num_labels++] = str[i];
    }
    return 0;
}

Tensor* tensor_einsum(const char* equation, Tensor** tensors, int num_tensors) {
    if (!equation || !tensors || num_tensors <= 0) return NULL;

    const char* arrow = strstr(equation, "->");
    if (!arrow) {
        LOG_ERROR("tensor_einsum: equation must contain '->'");
        return NULL;
    }

    const char* lhs = equation;
    int lhs_len = (int)(arrow - equation);
    const char* rhs = arrow + 2;
    int rhs_len = (int)strlen(rhs);

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
        if (parse_einsum_side(input_strs[i], input_lens[i], &inputs[i]) != 0) return NULL;
        if (inputs[i].num_labels != tensors[i]->ndim) {
            LOG_ERROR("tensor_einsum: operand %d has %d dims but subscript has %d labels",
                      i, tensors[i]->ndim, inputs[i].num_labels);
            return NULL;
        }
    }

    if (parse_einsum_side(rhs, rhs_len, &output) != 0) return NULL;

    char all_labels[128];
    int label_sizes[128];
    int num_unique = 0;

    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < inputs[i].num_labels; j++) {
            char c = inputs[i].labels[j];
            int size = tensors[i]->shape[j];
            int found = -1;
            for (int k = 0; k < num_unique; k++) {
                if (all_labels[k] == c) { found = k; break; }
            }
            if (found >= 0) {
                if (label_sizes[found] != size) {
                    LOG_ERROR("tensor_einsum: inconsistent size for label '%c'", c);
                    return NULL;
                }
            } else {
                all_labels[num_unique] = c;
                label_sizes[num_unique] = size;
                num_unique++;
            }
        }
    }

    int out_ndim = output.num_labels;
    int out_shape[64];
    size_t out_numel = 1;
    for (int i = 0; i < out_ndim; i++) {
        int found = -1;
        for (int k = 0; k < num_unique; k++) {
            if (all_labels[k] == output.labels[i]) { found = k; break; }
        }
        if (found < 0) {
            LOG_ERROR("tensor_einsum: output label '%c' not in inputs", output.labels[i]);
            return NULL;
        }
        out_shape[i] = label_sizes[found];
        out_numel *= (size_t)out_shape[i];
    }

    if (out_ndim == 0) {
        out_ndim = 1;
        out_shape[0] = 1;
        out_numel = 1;
    }

    for (int i = 0; i < num_tensors; i++) {
        tensor_ensure_executed(tensors[i]);
        if (!tensors[i]->data) return NULL;
    }

    TensorConfig config = {
        .dtype = tensors[0]->dtype, .device = tensors[0]->device,
        .has_dtype = true, .has_device = true
    };
    Tensor* result = tensor_zeros(out_shape, out_ndim, &config);
    if (!result) return NULL;

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
            size_t flat = 0;
            size_t stride = 1;
            for (int d = inputs[t].num_labels - 1; d >= 0; d--) {
                char c = inputs[t].labels[d];
                int k;
                for (k = 0; k < num_unique; k++) {
                    if (all_labels[k] == c) break;
                }
                flat += (size_t)label_indices[k] * stride;
                stride *= (size_t)tensors[t]->shape[d];
            }
            product *= tensor_get_float(tensors[t], flat);
        }

        size_t out_flat = 0;
        size_t out_stride = 1;
        bool scalar_output = (output.num_labels == 0);
        if (!scalar_output) {
            for (int d = output.num_labels - 1; d >= 0; d--) {
                char c = output.labels[d];
                int k;
                for (k = 0; k < num_unique; k++) {
                    if (all_labels[k] == c) break;
                }
                out_flat += (size_t)label_indices[k] * out_stride;
                out_stride *= (size_t)out_shape[d];
            }
        }

        out_data[out_flat] += product;

        for (int k = num_unique - 1; k >= 0; k--) {
            label_indices[k]++;
            if (label_indices[k] < label_sizes[k]) break;
            label_indices[k] = 0;
        }
    }

    if (output.num_labels == 0) {
        int scalar_shape[] = {1};
        Tensor* scalar = tensor_full(scalar_shape, 1, &config, out_data[0]);
        tensor_free(result);
        return scalar;
    }

    return result;
}
