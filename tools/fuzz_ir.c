/*
 * fuzz_ir.c — Structure-aware libFuzzer / AFL entry point for the C-ML IR.
 *
 * Build with:  cmake -DBUILD_FUZZ=ON -DCMAKE_C_COMPILER=clang ..
 *              make fuzz_ir
 * Run:         ./build/bin/fuzz_ir corpus/
 *
 * With AFL:    AFL_SKIP_CPUFREQ=1 afl-fuzz -i corpus/ -o findings/ -- ./build/bin/fuzz_ir @@
 *
 * The fuzzer interprets an input byte stream as a sequence of typed commands
 * that build an IR computation graph. Commands are kept simple so the fuzzer
 * can mutate them meaningfully:
 *
 *   byte 0: command  (0=new_tensor 1=relu 2=sigmoid 3=tanh 4=add 5=mul
 *                      6=matmul 7=realize 8=free 9=reset 10=linear ...)
 *   bytes 1+: args   (interpreted per command)
 */

#include "cml.h"
#include "alloc/cml_allocator.h"
#include "ops/ir/context.h"
#include "tensor/realize.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

/* ── fuzzer corpus: parse bytes into commands ─────────────────────────────── */

#define MAX_SLOTS 8

typedef struct {
    const uint8_t* data;
    size_t         len;
    size_t         pos;
} Reader;

static uint8_t r_u8(Reader* r) {
    return r->pos < r->len ? r->data[r->pos++] : 0;
}

static uint16_t r_u16(Reader* r) {
    uint16_t v = 0;
    memcpy(&v, r->data + (r->pos < r->len ? r->pos : r->len - 1),
           (r->pos + 2 <= r->len) ? 2 : 0);
    r->pos += 2;
    return v;
}

static int r_dim(Reader* r) {
    return (int)(r_u8(r) % 8) + 1; /* 1..8 */
}

/* ── main fuzzer entry ───────────────────────────────────────────────────── */

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;

    /* Suppress prints from CML internals during fuzzing */
    cml_ir_reset_global_context();

    Tensor* slots[MAX_SLOTS] = {0};

    Reader r = {data, size, 0};

    while (r.pos < r.len) {
        uint8_t cmd = r_u8(&r) % 16;
        uint8_t ai  = r_u8(&r) % MAX_SLOTS;
        uint8_t bi  = r_u8(&r) % MAX_SLOTS;

        switch (cmd) {
        case 0: { /* new_tensor: random small shape, float32 */
            int ndim = (r_u8(&r) % 3) + 1;
            int shape[3];
            for (int d = 0; d < ndim; d++) shape[d] = (r_u8(&r) % 4) + 1;
            int total = 1;
            for (int d = 0; d < ndim; d++) total *= shape[d];
            float* buf = calloc((size_t)total, sizeof(float));
            if (!buf) break;
            for (int j = 0; j < total; j++) buf[j] = (float)(r_u8(&r)) / 127.5f - 1.0f;
            TensorConfig cfg = {.dtype=DTYPE_FLOAT32,.device=DEVICE_CPU,
                                .has_dtype=true,.has_device=true};
            Tensor* t = cml_tensor(buf, shape, ndim, &cfg);
            free(buf);
            if (t) tensor_realize(t);
            if (slots[ai]) tensor_free(slots[ai]);
            slots[ai] = t;
            break;
        }
        case 1: /* relu */
            if (slots[ai]) {
                Tensor* out = tensor_relu(slots[ai]);
                if (slots[bi]) tensor_free(slots[bi]);
                slots[bi] = out;
            }
            break;
        case 2: /* sigmoid */
            if (slots[ai]) {
                Tensor* out = tensor_sigmoid(slots[ai]);
                if (slots[bi]) tensor_free(slots[bi]);
                slots[bi] = out;
            }
            break;
        case 3: /* tanh */
            if (slots[ai]) {
                Tensor* out = tensor_tanh(slots[ai]);
                if (slots[bi]) tensor_free(slots[bi]);
                slots[bi] = out;
            }
            break;
        case 4: /* add (shapes must match) */
            if (slots[ai] && slots[bi] && slots[ai]->ndim == slots[bi]->ndim) {
                bool ok = true;
                for (int d = 0; d < slots[ai]->ndim; d++)
                    if (slots[ai]->shape[d] != slots[bi]->shape[d]) { ok = false; break; }
                if (ok) {
                    uint8_t oi = r_u8(&r) % MAX_SLOTS;
                    Tensor* out = tensor_add(slots[ai], slots[bi]);
                    if (slots[oi]) tensor_free(slots[oi]);
                    slots[oi] = out;
                }
            }
            break;
        case 5: /* mul */
            if (slots[ai] && slots[bi] && slots[ai]->ndim == slots[bi]->ndim) {
                bool ok = true;
                for (int d = 0; d < slots[ai]->ndim; d++)
                    if (slots[ai]->shape[d] != slots[bi]->shape[d]) { ok = false; break; }
                if (ok) {
                    uint8_t oi = r_u8(&r) % MAX_SLOTS;
                    Tensor* out = tensor_mul(slots[ai], slots[bi]);
                    if (slots[oi]) tensor_free(slots[oi]);
                    slots[oi] = out;
                }
            }
            break;
        case 6: /* matmul: a[...,M,K] @ b[K,N] */
            if (slots[ai] && slots[bi] && slots[ai]->ndim >= 2 && slots[bi]->ndim == 2 &&
                slots[ai]->shape[slots[ai]->ndim-1] == slots[bi]->shape[0]) {
                uint8_t oi = r_u8(&r) % MAX_SLOTS;
                Tensor* out = tensor_matmul(slots[ai], slots[bi]);
                if (slots[oi]) tensor_free(slots[oi]);
                slots[oi] = out;
            }
            break;
        case 7: /* realize */
            if (slots[ai]) tensor_realize(slots[ai]);
            break;
        case 8: /* free slot */
            if (slots[ai]) { tensor_free(slots[ai]); slots[ai] = NULL; }
            break;
        case 9: /* ir reset (safe mid-stream) */
            cml_ir_reset_global_context();
            break;
        case 10: /* backward on slot (requires_grad) */
            if (slots[ai]) {
                slots[ai]->requires_grad = true;
                Tensor* ones = tensor_ones_like(slots[ai]);
                if (ones) {
                    tensor_backward(slots[ai], ones, false, false);
                    tensor_free(ones);
                }
            }
            break;
        case 11: /* copy slot */
            if (slots[ai]) {
                slots[ai]->ref_count++;
                if (slots[bi]) tensor_free(slots[bi]);
                slots[bi] = slots[ai];
            }
            break;
        default:
            break;
        }
    }

    /* clean up */
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (slots[i]) tensor_free(slots[i]);
    }
    cml_ir_reset_global_context();
    return 0;
}

/*
 * When NOT built with -fsanitize=fuzzer, provide a minimal AFL-compatible
 * main that reads stdin or a file argument.
 */
#ifndef __AFL_FUZZ_TESTCASE_LEN

#include <stdio.h>

int main(int argc, char** argv) {
    cml_init();

    FILE* f = (argc > 1) ? fopen(argv[1], "rb") : stdin;
    if (!f) { fprintf(stderr, "cannot open input\n"); return 1; }

    uint8_t buf[65536];
    size_t n = fread(buf, 1, sizeof(buf), f);
    if (argc > 1) fclose(f);

    int ret = LLVMFuzzerTestOneInput(buf, n);
    cml_cleanup();
    return ret;
}

#endif
