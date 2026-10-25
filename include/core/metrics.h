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

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_METRICS_H */
