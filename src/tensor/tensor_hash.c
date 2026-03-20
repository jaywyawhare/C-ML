#include "tensor/tensor.h"
#include <stdlib.h>
#include <string.h>

#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME        0x100000001b3ULL

uint64_t tensor_hash(Tensor* t) {
    if (!t) return 0;
    tensor_ensure_executed(t);
    if (!t->data) return 0;

    size_t nbytes = t->numel * cml_dtype_size(t->dtype);
    const uint8_t* bytes = (const uint8_t*)t->data;

    uint64_t h = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < nbytes; i++) {
        h ^= bytes[i];
        h *= FNV_PRIME;
    }
    return h;
}

/* Keccak-f[1600] / SHA3-256 */

static const uint64_t keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

static const int keccak_rot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

static const int keccak_pi[25] = {
     0, 10, 20,  5, 15,
    16,  1, 11, 21,  6,
     7, 17,  2, 12, 22,
    23,  8, 18,  3, 13,
    14, 24,  9, 19,  4,
};

static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

static void keccak_f1600(uint64_t state[25]) {
    for (int round = 0; round < 24; round++) {
        /* theta */
        uint64_t c[5];
        for (int x = 0; x < 5; x++)
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];

        uint64_t d[5];
        for (int x = 0; x < 5; x++)
            d[x] = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);

        for (int i = 0; i < 25; i++)
            state[i] ^= d[i % 5];

        /* rho + pi */
        uint64_t tmp[25];
        for (int i = 0; i < 25; i++)
            tmp[keccak_pi[i]] = rotl64(state[i], keccak_rot[i]);

        /* chi */
        for (int y = 0; y < 5; y++) {
            int base = y * 5;
            for (int x = 0; x < 5; x++)
                state[base + x] = tmp[base + x] ^ (~tmp[base + (x + 1) % 5] & tmp[base + (x + 2) % 5]);
        }

        /* iota */
        state[0] ^= keccak_rc[round];
    }
}

#define SHA3_256_RATE 136

int tensor_keccak(Tensor* t, uint8_t* out, size_t out_len) {
    if (!t || !out || out_len == 0) return -1;
    tensor_ensure_executed(t);
    if (!t->data) return -1;

    size_t nbytes = t->numel * cml_dtype_size(t->dtype);
    const uint8_t* input = (const uint8_t*)t->data;

    if (out_len > 32) out_len = 32;

    uint64_t state[25];
    memset(state, 0, sizeof(state));
    uint8_t* state_bytes = (uint8_t*)state;

    /* absorb */
    size_t offset = 0;
    while (offset + SHA3_256_RATE <= nbytes) {
        for (size_t i = 0; i < SHA3_256_RATE; i++)
            state_bytes[i] ^= input[offset + i];
        keccak_f1600(state);
        offset += SHA3_256_RATE;
    }

    /* final block with SHA3 padding: 0x06 ... 0x80 */
    size_t remaining = nbytes - offset;
    uint8_t pad[SHA3_256_RATE];
    memset(pad, 0, SHA3_256_RATE);
    if (remaining > 0)
        memcpy(pad, input + offset, remaining);

    pad[remaining] = 0x06;
    pad[SHA3_256_RATE - 1] |= 0x80;

    for (size_t i = 0; i < SHA3_256_RATE; i++)
        state_bytes[i] ^= pad[i];
    keccak_f1600(state);

    /* squeeze */
    memcpy(out, state_bytes, out_len);
    return 0;
}
