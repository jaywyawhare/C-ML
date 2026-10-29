/*
 * pack_padded_sequence / pad_packed_sequence. Oracles: hand-computed batch_sizes
 * and packed length for a known ragged batch; a pack->pad round-trip that must
 * reproduce the input at valid positions; and an RNN-cell padding-invariance
 * check - driving a cell over the packed batch must equal running it per sequence
 * on the unpadded inputs, which is the whole point of packing.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "core/threefry.h"
#include "nn/layers/rnn.h"
#include "nn/packed_sequence.h"
#include "test_harness.h"

#define B 3
#define T 4
#define F 2
#define H 3

static TensorConfig cfg = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static const int LENGTHS[B] = {4, 2, 3};

/* [B,T,F] batch_first input; value(b,t,f) = 100*b + 10*t + f, padding rows zeroed. */
static Tensor* make_input(void) {
    int sh[3] = {B, T, F};
    Tensor* t = tensor_zeros(sh, 3, &cfg);
    float* d  = (float*)tensor_data_ptr(t);
    for (int b = 0; b < B; b++)
        for (int tt = 0; tt < T; tt++)
            for (int f = 0; f < F; f++)
                d[(b * T + tt) * F + f] = (tt < LENGTHS[b]) ? (100.0f * b + 10.0f * tt + f) : 0.0f;
    return t;
}

static int test_pack_batch_sizes(void) {
    cml_reset_ir_context();
    Tensor* in           = make_input();
    CMLPackedSequence* p = cml_pack_padded_sequence(in, LENGTHS, B, true, false);
    if (!p) {
        cml_reset_ir_context();
        return 0;
    }
    int want_bs[4] = {3, 3, 2, 1};                              /* sorted lengths 4,3,2 */
    int ok         = p->max_len == 4 && p->data->shape[0] == 9; /* sum(lengths)=9 */
    for (int t = 0; t < 4 && ok; t++)
        if (p->batch_sizes[t] != want_bs[t])
            ok = 0;
    /* sorted order is sequence 0 (len4), 2 (len3), 1 (len2). */
    ok = ok && p->sorted_indices[0] == 0 && p->sorted_indices[1] == 2 && p->sorted_indices[2] == 1;
    cml_packed_sequence_free(p);
    cml_reset_ir_context();
    return ok;
}

static int test_roundtrip(void) {
    cml_reset_ir_context();
    Tensor* in           = make_input();
    float* ind           = (float*)tensor_data_ptr(in);
    CMLPackedSequence* p = cml_pack_padded_sequence(in, LENGTHS, B, true, false);
    int lens_out[B]      = {0};
    Tensor* out          = cml_pad_packed_sequence(p, true, 0.0f, 0, lens_out);
    if (!out || tensor_ensure_executed(out) != 0) {
        cml_reset_ir_context();
        return 0;
    }
    float* od = (float*)tensor_data_ptr(out);
    int ok    = 1;
    for (int b = 0; b < B && ok; b++) {
        if (lens_out[b] != LENGTHS[b])
            ok = 0;
        for (int tt = 0; tt < T && ok; tt++)
            for (int f = 0; f < F; f++)
                if (fabsf(od[(b * T + tt) * F + f] - ind[(b * T + tt) * F + f]) > 1e-6f)
                    ok = 0;
    }
    cml_packed_sequence_free(p);
    cml_reset_ir_context();
    return ok;
}

/* Extract one [1,F] row from the packed data buffer. */
static Tensor* packed_row(const float* data, int row) {
    int sh[2] = {1, F};
    Tensor* t = tensor_zeros(sh, 2, &cfg);
    memcpy(tensor_data_ptr(t), data + (size_t)row * F, F * sizeof(float));
    return t;
}

static int test_rnn_padding_invariance(void) {
    cml_reset_ir_context();
    cml_rng_set_global_seed(42);
    RNNCell* cell = nn_rnn_cell(F, H, true, DTYPE_FLOAT32, DEVICE_CPU);
    Tensor* in    = make_input();
    float* ind    = (float*)tensor_data_ptr(in);

    CMLPackedSequence* p = cml_pack_padded_sequence(in, LENGTHS, B, true, false);
    float* pd            = (float*)tensor_data_ptr(p->data);

    /* Packed run: one hidden state per sorted sequence, stepped only while active. */
    Tensor* hs[B] = {0};
    int row       = 0;
    for (int t = 0; t < p->max_len; t++)
        for (int pos = 0; pos < p->batch_sizes[t]; pos++) {
            hs[pos] = rnn_cell_forward(cell, packed_row(pd, row), hs[pos]);
            row++;
        }

    int ok = 1;
    for (int pos = 0; pos < B && ok; pos++) {
        int b = p->sorted_indices[pos];
        /* Reference: run the same cell over sequence b's unpadded inputs. */
        Tensor* href = NULL;
        for (int t = 0; t < LENGTHS[b]; t++) {
            int sh[2]  = {1, F};
            Tensor* xr = tensor_zeros(sh, 2, &cfg);
            memcpy(tensor_data_ptr(xr), ind + (size_t)(b * T + t) * F, F * sizeof(float));
            href = rnn_cell_forward(cell, xr, href);
        }
        tensor_ensure_executed(hs[pos]);
        tensor_ensure_executed(href);
        float* a = (float*)tensor_data_ptr(hs[pos]);
        float* r = (float*)tensor_data_ptr(href);
        for (int i = 0; i < H; i++)
            if (fabsf(a[i] - r[i]) > 1e-5f)
                ok = 0;
    }
    cml_packed_sequence_free(p);
    cml_reset_ir_context();
    return ok;
}

int main(void) {
    printf("test_packed_sequence\n\n");
    TEST(pack_batch_sizes);
    TEST(roundtrip);
    TEST(rnn_padding_invariance);
    cml_reset_ir_context();
    return TEST_SUMMARY();
}
