#include "nn/sampling.h"
#include "nn/llm_ops.h"
#include "alloc/cml_allocator.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
    float v;
    int i;
} IdxVal;

/** Descending comparator on value (ties broken by ascending index for stability). */
static int idxval_desc(const void* a, const void* b) {
    const IdxVal* x = (const IdxVal*)a;
    const IdxVal* y = (const IdxVal*)b;
    if (x->v < y->v)
        return 1;
    if (x->v > y->v)
        return -1;
    return x->i - y->i;
}

/** Ascending comparator on value (ties broken by ascending index for stability). */
static int idxval_asc(const void* a, const void* b) {
    const IdxVal* x = (const IdxVal*)a;
    const IdxVal* y = (const IdxVal*)b;
    if (x->v < y->v)
        return -1;
    if (x->v > y->v)
        return 1;
    return x->i - y->i;
}

/**
 * Keep the smallest prefix of a sorted order whose cumulative probability reaches threshold, mask
 * the rest to -inf. Returns kept count.
 * @param logits (float*)
 * @param probs (const float*)
 * @param sorted (const IdxVal*)
 * @param n (int)
 * @param threshold (float)
 * @return (int)
 */
static int mask_cumprob_tail(float* logits, const float* probs, const IdxVal* sorted, int n,
                             float threshold) {
    float cum = 0.0f;
    int cut   = n;
    for (int r = 0; r < n; r++) {
        cum += probs[sorted[r].i];
        if (cum >= threshold) {
            cut = r + 1;
            break;
        }
    }
    int kept = 0;
    for (int r = 0; r < n; r++) {
        if (r >= cut)
            logits[sorted[r].i] = -INFINITY;
        else
            kept++;
    }
    return kept;
}

void cml_logits_temperature(float* logits, int n, float temperature) {
    if (!logits || n <= 0 || temperature <= 0.0f || temperature == 1.0f)
        return;
    float inv = 1.0f / temperature;
    for (int i = 0; i < n; i++)
        logits[i] *= inv;
}

void cml_logits_repetition_penalty(float* logits, int n, const int* tokens, int num_tokens,
                                   float penalty) {
    if (!logits || n <= 0 || !tokens || num_tokens <= 0 || penalty <= 0.0f || penalty == 1.0f)
        return;
    for (int t = 0; t < num_tokens; t++) {
        int id = tokens[t];
        if (id < 0 || id >= n)
            continue;
        logits[id] = (logits[id] > 0.0f) ? logits[id] / penalty : logits[id] * penalty;
    }
}

int cml_logits_top_k(float* logits, int n, int k) {
    if (!logits || n <= 0)
        return 0;
    if (k <= 0 || k >= n)
        return n;
    IdxVal* iv = (IdxVal*)cml_malloc((size_t)n * sizeof(IdxVal));
    if (!iv)
        return n;
    for (int i = 0; i < n; i++) {
        iv[i].v = logits[i];
        iv[i].i = i;
    }
    qsort(iv, (size_t)n, sizeof(IdxVal), idxval_desc);
    /* Keep everything >= the k-th largest value (ties at the threshold stay, matching HF). */
    float threshold = iv[k - 1].v;
    cml_free(iv);
    int kept = 0;
    for (int i = 0; i < n; i++) {
        if (logits[i] < threshold)
            logits[i] = -INFINITY;
        else
            kept++;
    }
    return kept;
}

int cml_logits_top_p(float* logits, int n, float top_p) {
    if (!logits || n <= 0)
        return 0;
    if (top_p <= 0.0f || top_p >= 1.0f)
        return n;

    float* probs = (float*)cml_malloc((size_t)n * sizeof(float));
    IdxVal* iv   = (IdxVal*)cml_malloc((size_t)n * sizeof(IdxVal));
    if (!probs || !iv) {
        cml_free(probs);
        cml_free(iv);
        return n;
    }
    cml_softmax_row_temp(logits, n, 1.0f, probs);
    for (int i = 0; i < n; i++) {
        iv[i].v = logits[i];
        iv[i].i = i;
    }
    qsort(iv, (size_t)n, sizeof(IdxVal), idxval_desc);
    int kept = mask_cumprob_tail(logits, probs, iv, n, top_p);
    cml_free(probs);
    cml_free(iv);
    return kept;
}

int cml_logits_min_p(float* logits, int n, float min_p) {
    if (!logits || n <= 0)
        return 0;
    if (min_p <= 0.0f)
        return n;

    float* probs = (float*)cml_malloc((size_t)n * sizeof(float));
    if (!probs)
        return n;
    cml_softmax_row_temp(logits, n, 1.0f, probs);
    float max_prob = 0.0f;
    for (int i = 0; i < n; i++)
        if (probs[i] > max_prob)
            max_prob = probs[i];

    float threshold = min_p * max_prob;
    int kept        = 0;
    for (int i = 0; i < n; i++) {
        if (probs[i] < threshold)
            logits[i] = -INFINITY;
        else
            kept++;
    }
    cml_free(probs);
    return kept;
}

void cml_logits_bias(float* logits, int n, const int* ids, const float* biases, int count) {
    if (!logits || n <= 0 || !ids || !biases || count <= 0)
        return;
    for (int k = 0; k < count; k++) {
        int id = ids[k];
        if (id < 0 || id >= n)
            continue;
        logits[id] += biases[k];
    }
}

void cml_logits_frequency_presence_penalty(float* logits, int n, const int* history, int hist_len,
                                           float frequency_penalty, float presence_penalty) {
    if (!logits || n <= 0 || !history || hist_len <= 0)
        return;
    if (frequency_penalty == 0.0f && presence_penalty == 0.0f)
        return;
    /* `seen` flips to 1 on a token's first occurrence so presence is charged exactly once. */
    unsigned char* seen = (unsigned char*)cml_calloc((size_t)n, sizeof(unsigned char));
    if (!seen)
        return;
    for (int t = 0; t < hist_len; t++) {
        int id = history[t];
        if (id < 0 || id >= n)
            continue;
        logits[id] -= frequency_penalty;
        if (!seen[id]) {
            logits[id] -= presence_penalty;
            seen[id] = 1;
        }
    }
    cml_free(seen);
}

int cml_logits_no_repeat_ngram(float* logits, int n, const int* history, int hist_len,
                               int ngram_size) {
    if (!logits || n <= 0)
        return n;
    if (ngram_size <= 0 || !history || hist_len < ngram_size)
        return n;
    int prefix_len    = ngram_size - 1;
    const int* prefix = history + hist_len - prefix_len; /* trailing (ngram_size-1) tokens */
    int banned        = 0;
    /* Scan every position whose n-gram could have been completed earlier in the history. */
    for (int i = 0; i + ngram_size <= hist_len; i++) {
        int match = 1;
        for (int j = 0; j < prefix_len; j++)
            if (history[i + j] != prefix[j]) {
                match = 0;
                break;
            }
        if (match) {
            int tok = history[i + prefix_len];
            if (tok >= 0 && tok < n && logits[tok] != -INFINITY) {
                logits[tok] = -INFINITY;
                banned++;
            }
        }
    }
    return n - banned;
}

int cml_logits_typical(float* logits, int n, float typical_p) {
    if (!logits || n <= 0)
        return 0;
    if (typical_p <= 0.0f || typical_p >= 1.0f)
        return n;

    float* probs = (float*)cml_malloc((size_t)n * sizeof(float));
    IdxVal* iv   = (IdxVal*)cml_malloc((size_t)n * sizeof(IdxVal));
    if (!probs || !iv) {
        cml_free(probs);
        cml_free(iv);
        return n;
    }
    cml_softmax_row_temp(logits, n, 1.0f, probs);
    float entropy = 0.0f;
    for (int i = 0; i < n; i++)
        if (probs[i] > 0.0f)
            entropy -= probs[i] * logf(probs[i]);
    /* Rank by absolute deviation of each token's surprisal (-log p) from the entropy. */
    for (int i = 0; i < n; i++) {
        float surprisal = (probs[i] > 0.0f) ? -logf(probs[i]) : 1e30f;
        iv[i].v         = fabsf(surprisal - entropy);
        iv[i].i         = i;
    }
    qsort(iv, (size_t)n, sizeof(IdxVal), idxval_asc);
    int kept = mask_cumprob_tail(logits, probs, iv, n, typical_p);
    cml_free(probs);
    cml_free(iv);
    return kept;
}
