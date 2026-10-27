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

    /* Smallest descending prefix whose cumulative probability reaches top_p; mask the rest. */
    float cum = 0.0f;
    int cut   = n; /* first index (in sorted order) to drop */
    for (int r = 0; r < n; r++) {
        cum += probs[iv[r].i];
        if (cum >= top_p) {
            cut = r + 1;
            break;
        }
    }
    int kept = 0;
    for (int r = 0; r < n; r++) {
        if (r >= cut)
            logits[iv[r].i] = -INFINITY;
        else
            kept++;
    }
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
