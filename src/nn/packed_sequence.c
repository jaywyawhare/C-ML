#include "nn/packed_sequence.h"

#include <string.h>

#include "alloc/cml_allocator.h"
#include "cml.h"

CMLPackedSequence* cml_pack_padded_sequence(Tensor* input, const int* lengths, int batch_size,
                                            bool batch_first, bool enforce_sorted) {
    if (!input || !lengths || batch_size <= 0 || input->ndim != 3)
        return NULL;
    int B = batch_size;
    int T = batch_first ? input->shape[1] : input->shape[0];
    int F = input->shape[2];
    if ((batch_first ? input->shape[0] : input->shape[1]) != B)
        return NULL;
    if (tensor_ensure_executed(input) != 0)
        return NULL;
    const float* in = (const float*)tensor_data_ptr(input);
    if (!in)
        return NULL;

    CMLPackedSequence* p = (CMLPackedSequence*)cml_malloc(sizeof(CMLPackedSequence));
    if (!p)
        return NULL;
    memset(p, 0, sizeof(*p));
    p->batch_size       = B;
    p->feature_size     = F;
    p->sorted_indices   = (int*)cml_malloc((size_t)B * sizeof(int));
    p->unsorted_indices = (int*)cml_malloc((size_t)B * sizeof(int));
    if (!p->sorted_indices || !p->unsorted_indices) {
        cml_packed_sequence_free(p);
        return NULL;
    }

    for (int i = 0; i < B; i++)
        p->sorted_indices[i] = i;
    if (!enforce_sorted) {
        /* Stable descending sort of indices by length (insertion sort, B is small). */
        for (int i = 1; i < B; i++) {
            int key = p->sorted_indices[i];
            int j   = i - 1;
            while (j >= 0 && lengths[p->sorted_indices[j]] < lengths[key]) {
                p->sorted_indices[j + 1] = p->sorted_indices[j];
                j--;
            }
            p->sorted_indices[j + 1] = key;
        }
    }
    for (int pos = 0; pos < B; pos++)
        p->unsorted_indices[p->sorted_indices[pos]] = pos;

    int max_len = 0, total = 0;
    for (int i = 0; i < B; i++) {
        if (lengths[i] > max_len)
            max_len = lengths[i];
        total += lengths[i];
    }
    p->max_len     = max_len;
    p->batch_sizes = (int*)cml_malloc((size_t)(max_len > 0 ? max_len : 1) * sizeof(int));
    if (!p->batch_sizes) {
        cml_packed_sequence_free(p);
        return NULL;
    }
    for (int t = 0; t < max_len; t++) {
        int cnt = 0;
        for (int pos = 0; pos < B; pos++)
            if (lengths[p->sorted_indices[pos]] > t)
                cnt++;
        p->batch_sizes[t] = cnt;
    }

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = input->device, .has_dtype = true, .has_device = true};
    int dsh[2] = {total, F};
    p->data    = cml_zeros(dsh, 2, &cfg);
    if (!p->data) {
        cml_packed_sequence_free(p);
        return NULL;
    }
    float* dd = (float*)tensor_data_ptr(p->data);
    int row   = 0;
    for (int t = 0; t < max_len; t++) {
        for (int pos = 0; pos < p->batch_sizes[t]; pos++) {
            int orig       = p->sorted_indices[pos];
            size_t src_off = batch_first ? ((size_t)orig * T + t) * F : ((size_t)t * B + orig) * F;
            memcpy(dd + (size_t)row * F, in + src_off, (size_t)F * sizeof(float));
            row++;
        }
    }
    return p;
}

Tensor* cml_pad_packed_sequence(const CMLPackedSequence* packed, bool batch_first,
                                float padding_value, int total_length, int* out_lengths) {
    if (!packed || !packed->data)
        return NULL;
    int B = packed->batch_size;
    int F = packed->feature_size;
    int T = (total_length > 0) ? total_length : packed->max_len;
    if (T < packed->max_len)
        return NULL;
    if (tensor_ensure_executed(packed->data) != 0)
        return NULL;
    const float* dd = (const float*)tensor_data_ptr(packed->data);

    TensorConfig cfg = {.dtype      = DTYPE_FLOAT32,
                        .device     = packed->data->device,
                        .has_dtype  = true,
                        .has_device = true};
    int osh[3]       = {batch_first ? B : T, batch_first ? T : B, F};
    Tensor* out      = cml_full(osh, 3, &cfg, padding_value);
    if (!out)
        return NULL;
    float* od = (float*)tensor_data_ptr(out);

    int row = 0;
    for (int t = 0; t < packed->max_len; t++) {
        for (int pos = 0; pos < packed->batch_sizes[t]; pos++) {
            int orig       = packed->sorted_indices[pos];
            size_t dst_off = batch_first ? ((size_t)orig * T + t) * F : ((size_t)t * B + orig) * F;
            memcpy(od + dst_off, dd + (size_t)row * F, (size_t)F * sizeof(float));
            row++;
        }
    }

    if (out_lengths) {
        for (int pos = 0; pos < B; pos++) {
            int len = 0;
            for (int t = 0; t < packed->max_len; t++)
                if (packed->batch_sizes[t] > pos)
                    len++;
            out_lengths[packed->sorted_indices[pos]] = len;
        }
    }
    return out;
}

void cml_packed_sequence_free(CMLPackedSequence* p) {
    if (!p)
        return;
    if (p->data)
        tensor_free(p->data);
    cml_free(p->batch_sizes);
    cml_free(p->sorted_indices);
    cml_free(p->unsorted_indices);
    cml_free(p);
}
