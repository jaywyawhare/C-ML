#ifndef CML_CORE_METRICS_H
#define CML_CORE_METRICS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Evaluation metrics over already-decoded predictions. Classification metrics take
 * integer class labels in [0, num_classes); regression metrics take float vectors.
 * All are pure, deterministic and allocation-light so they are hand-verifiable. */

/* Row-major num_classes x num_classes count matrix, row = true class, col = predicted. */
typedef struct CMLConfusionMatrix {
    int num_classes;
    int* counts;
} CMLConfusionMatrix;

/**
 * Build a confusion matrix from n (true, predicted) label pairs.
 * @param y_true (const int*)
 * @param y_pred (const int*)
 * @param n (int)
 * @param num_classes (int)
 * @return (CMLConfusionMatrix*)
 */
CMLConfusionMatrix* cml_confusion_matrix(const int* y_true, const int* y_pred, int n,
                                         int num_classes);

/**
 * Free a confusion matrix.
 * @param cm (CMLConfusionMatrix*)
 * @return (void)
 */
void cml_confusion_matrix_free(CMLConfusionMatrix* cm);

/**
 * Fraction of exact matches between n true and predicted labels.
 * @param y_true (const int*)
 * @param y_pred (const int*)
 * @param n (int)
 * @return (float)
 */
float cml_accuracy(const int* y_true, const int* y_pred, int n);

/* Averaging for multi-class precision/recall/F1. PER_CLASS writes one value per class
 * into the caller's out[] and returns their unweighted mean. */
typedef enum {
    CML_AVG_MACRO,    /* unweighted mean over classes */
    CML_AVG_MICRO,    /* global tally (equals accuracy for single-label) */
    CML_AVG_WEIGHTED, /* mean weighted by each class's true-sample count */
    CML_AVG_PER_CLASS,
} CMLAverage;

/**
 * Precision = TP / (TP + FP), averaged per @p average. If @p per_class is non-NULL it
 * receives num_classes per-class values.
 * @param cm (const CMLConfusionMatrix*)
 * @param average (CMLAverage)
 * @param per_class (float*)
 * @return (float)
 */
float cml_precision(const CMLConfusionMatrix* cm, CMLAverage average, float* per_class);

/**
 * Recall = TP / (TP + FN), averaged per @p average.
 * @param cm (const CMLConfusionMatrix*)
 * @param average (CMLAverage)
 * @param per_class (float*)
 * @return (float)
 */
float cml_recall(const CMLConfusionMatrix* cm, CMLAverage average, float* per_class);

/**
 * F1 = harmonic mean of precision and recall, averaged per @p average.
 * @param cm (const CMLConfusionMatrix*)
 * @param average (CMLAverage)
 * @param per_class (float*)
 * @return (float)
 */
float cml_f1_score(const CMLConfusionMatrix* cm, CMLAverage average, float* per_class);

/**
 * Binary ROC AUC from scores for the positive class (label 1), by the rank-sum
 * (Mann-Whitney U) identity; ties get averaged ranks. Returns 0.5 if a class is absent.
 * @param y_true (const int*)
 * @param y_score (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_roc_auc(const int* y_true, const float* y_score, int n);

/**
 * Coefficient of determination R^2 = 1 - SS_res / SS_tot.
 * @param y_true (const float*)
 * @param y_pred (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_r2_score(const float* y_true, const float* y_pred, int n);

/**
 * Root mean squared error.
 * @param y_true (const float*)
 * @param y_pred (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_rmse(const float* y_true, const float* y_pred, int n);

/**
 * Mean absolute error.
 * @param y_true (const float*)
 * @param y_pred (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_mae_metric(const float* y_true, const float* y_pred, int n);

/**
 * Top-k accuracy: fraction of samples whose true class is among the k highest logits.
 * @p logits is row-major [n_samples, n_classes].
 * @param logits (const float*)
 * @param n_samples (int)
 * @param n_classes (int)
 * @param y_true (const int*)
 * @param k (int)
 * @return (float)
 */
float cml_top_k_accuracy(const float* logits, int n_samples, int n_classes, const int* y_true,
                         int k);

/**
 * Perplexity exp(mean(nll)) from per-token negative log-likelihoods (natural log).
 * @param nll (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_perplexity(const float* nll, int n);

/**
 * Intersection-over-union of two axis-aligned boxes [x1, y1, x2, y2] (x2>x1, y2>y1).
 * @param a (const float*)
 * @param b (const float*)
 * @return (float)
 */
float cml_iou(const float* a, const float* b);

/**
 * Average precision (area under the precision-recall curve) for one class from binary
 * labels and detection scores: AP = sum over detections of (R_i - R_{i-1}) * P_i.
 * @param y_true (const int*)
 * @param y_score (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_average_precision(const int* y_true, const float* y_score, int n);

/**
 * Sentence BLEU: brevity penalty times the geometric mean of clipped n-gram precisions
 * for n = 1..max_n, over token-id sequences. Returns 0 if any n-gram precision is 0.
 * @param candidate (const int*)
 * @param cand_len (int)
 * @param reference (const int*)
 * @param ref_len (int)
 * @param max_n (int)
 * @return (float)
 */
float cml_bleu(const int* candidate, int cand_len, const int* reference, int ref_len, int max_n);

/**
 * Levenshtein edit distance (unit-cost insert/delete/substitute) over token-id sequences.
 * @param a (const int*)
 * @param alen (int)
 * @param b (const int*)
 * @param blen (int)
 * @return (int)
 */
int cml_edit_distance(const int* a, int alen, const int* b, int blen);

/**
 * Word error rate: reference-to-hypothesis edit distance divided by reference length.
 * @param reference (const int*)
 * @param ref_len (int)
 * @param hypothesis (const int*)
 * @param hyp_len (int)
 * @return (float)
 */
float cml_wer(const int* reference, int ref_len, const int* hypothesis, int hyp_len);

/**
 * ROUGE-N recall: clipped candidate/reference n-gram overlap over reference n-gram count.
 * @param candidate (const int*)
 * @param cand_len (int)
 * @param reference (const int*)
 * @param ref_len (int)
 * @param n (int)
 * @return (float)
 */
float cml_rouge_n(const int* candidate, int cand_len, const int* reference, int ref_len, int n);

/**
 * Matthews correlation coefficient for binary (0/1) labels.
 * @param y_true (const int*)
 * @param y_pred (const int*)
 * @param n (int)
 * @return (float)
 */
float cml_matthews_corrcoef(const int* y_true, const int* y_pred, int n);

/**
 * Cohen's kappa agreement over multi-class labels.
 * @param y_true (const int*)
 * @param y_pred (const int*)
 * @param n (int)
 * @param num_classes (int)
 * @return (float)
 */
float cml_cohen_kappa(const int* y_true, const int* y_pred, int n, int num_classes);

/**
 * Pearson correlation coefficient between two sequences.
 * @param x (const float*)
 * @param y (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_pearson(const float* x, const float* y, int n);

/**
 * Spearman rank correlation between two sequences.
 * @param x (const float*)
 * @param y (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_spearman(const float* x, const float* y, int n);

/**
 * Expected calibration error over equal-width confidence bins.
 * @param confidences (const float*)
 * @param correct (const int*)
 * @param n (int)
 * @param n_bins (int)
 * @return (float)
 */
float cml_expected_calibration_error(const float* confidences, const int* correct, int n,
                                     int n_bins);

/**
 * Multi-class cross-entropy (log loss): mean -log of the clipped probability of the true class.
 * @param y_true (const int*)
 * @param y_prob (const float*)
 * @param n (int)
 * @param num_classes (int)
 * @return (float)
 */
float cml_log_loss(const int* y_true, const float* y_prob, int n, int num_classes);

/**
 * Balanced accuracy: the macro average of per-class recall over classes with support.
 * @param y_true (const int*)
 * @param y_pred (const int*)
 * @param n (int)
 * @param num_classes (int)
 * @return (float)
 */
float cml_balanced_accuracy(const int* y_true, const int* y_pred, int n, int num_classes);

/**
 * Macro-averaged Jaccard score (intersection over union) over classes with non-empty union.
 * @param y_true (const int*)
 * @param y_pred (const int*)
 * @param n (int)
 * @param num_classes (int)
 * @return (float)
 */
float cml_jaccard_score(const int* y_true, const int* y_pred, int n, int num_classes);

/**
 * Hamming loss: the fraction of labels that are misclassified (1 - accuracy).
 * @param y_true (const int*)
 * @param y_pred (const int*)
 * @param n (int)
 * @return (float)
 */
float cml_hamming_loss(const int* y_true, const int* y_pred, int n);

/**
 * Normalized discounted cumulative gain at rank k for a single ranked list.
 * @param y_score (const float*)
 * @param y_true_rel (const float*)
 * @param n (int)
 * @param k (int)
 * @return (float)
 */
float cml_ndcg(const float* y_score, const float* y_true_rel, int n, int k);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_METRICS_H */
