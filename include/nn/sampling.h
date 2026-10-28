#ifndef CML_NN_SAMPLING_H
#define CML_NN_SAMPLING_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Scale a logits row in place by 1/temperature (no-op for temperature <= 0 or == 1).
 * @param logits (float*)
 * @param n (int)
 * @param temperature (float)
 * @return (void)
 */
void cml_logits_temperature(float* logits, int n, float temperature);

/**
 * Apply a repetition penalty in place to every already-seen token id (HF convention: divide a
 * positive logit by penalty, multiply a negative one). Out-of-range ids are ignored.
 * @param logits (float*)
 * @param n (int)
 * @param tokens (const int*)
 * @param num_tokens (int)
 * @param penalty (float)
 * @return (void)
 */
void cml_logits_repetition_penalty(float* logits, int n, const int* tokens, int num_tokens,
                                   float penalty);

/**
 * Top-k filter in place: mask every logit below the k-th largest to -inf. Returns kept count.
 * @param logits (float*)
 * @param n (int)
 * @param k (int)
 * @return (int)
 */
int cml_logits_top_k(float* logits, int n, int k);

/**
 * Nucleus (top-p) filter in place: keep the smallest descending prefix whose cumulative softmax
 * probability reaches top_p, masking the rest to -inf. Returns kept count.
 * @param logits (float*)
 * @param n (int)
 * @param top_p (float)
 * @return (int)
 */
int cml_logits_top_p(float* logits, int n, float top_p);

/**
 * Min-p filter in place: mask every token whose softmax probability is below min_p * max_prob.
 * Returns kept count.
 * @param logits (float*)
 * @param n (int)
 * @param min_p (float)
 * @return (int)
 */
int cml_logits_min_p(float* logits, int n, float min_p);

/**
 * Add per-token biases in place: logits[ids[k]] += biases[k]. Out-of-range ids are ignored; a
 * bias of -inf hard-bans a token. No-op when count <= 0.
 * @param logits (float*)
 * @param n (int)
 * @param ids (const int*)
 * @param biases (const float*)
 * @param count (int)
 * @return (void)
 */
void cml_logits_bias(float* logits, int n, const int* ids, const float* biases, int count);

/**
 * Apply OpenAI-style additive penalties in place over the token history: subtract
 * frequency_penalty once per prior occurrence of a token and presence_penalty once per token that
 * appears at all. No-op when both penalties are zero or the history is empty.
 * @param logits (float*)
 * @param n (int)
 * @param history (const int*)
 * @param hist_len (int)
 * @param frequency_penalty (float)
 * @param presence_penalty (float)
 * @return (void)
 */
void cml_logits_frequency_presence_penalty(float* logits, int n, const int* history, int hist_len,
                                           float frequency_penalty, float presence_penalty);

/**
 * Ban in place every token that would complete an n-gram already present in the history: the
 * trailing (ngram_size-1) tokens form the prefix, and any token that followed that same prefix
 * earlier is masked to -inf. Returns kept count.
 * @param logits (float*)
 * @param n (int)
 * @param history (const int*)
 * @param hist_len (int)
 * @param ngram_size (int)
 * @return (int)
 */
int cml_logits_no_repeat_ngram(float* logits, int n, const int* history, int hist_len,
                               int ngram_size);

/**
 * Locally-typical filter in place: keep the smallest set of tokens, ordered by how close their
 * surprisal is to the distribution entropy, whose cumulative softmax probability reaches typical_p;
 * mask the rest to -inf. Returns kept count.
 * @param logits (float*)
 * @param n (int)
 * @param typical_p (float)
 * @return (int)
 */
int cml_logits_typical(float* logits, int n, float typical_p);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_SAMPLING_H */
