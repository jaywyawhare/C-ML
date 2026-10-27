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

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_SAMPLING_H */
