#ifndef CML_CORE_CTC_H
#define CML_CORE_CTC_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Connectionist Temporal Classification loss for one sequence, computed by the log-space
 * forward-backward algorithm. log_probs is the [T, C] row-major log-softmax matrix (T frames,
 * C classes including blank); targets is the [target_len] label sequence (no blanks). Returns
 * the negative log-likelihood (+inf if the alignment is infeasible). If grad is non-NULL it
 * receives the [T*C] gradient of the loss with respect to log_probs.
 * @param log_probs (const float*)
 * @param T (int)
 * @param C (int)
 * @param targets (const int*)
 * @param target_len (int)
 * @param blank (int)
 * @param grad (float*)
 * @return (float)
 */
float cml_ctc_loss(const float* log_probs, int T, int C, const int* targets, int target_len,
                   int blank, float* grad);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_CTC_H */
