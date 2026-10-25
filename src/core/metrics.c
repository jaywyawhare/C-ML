#include "core/metrics.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"
#include <math.h>
#include <stdlib.h>

/** Build a confusion matrix from n (true, predicted) label pairs; labels outside
 *  [0, num_classes) are skipped.
 * @param y_true (const int*)
 * @param y_pred (const int*)
 * @param n (int)
 * @param num_classes (int)
 * @return (CMLConfusionMatrix*)
 */
CMLConfusionMatrix* cml_confusion_matrix(const int* y_true, const int* y_pred, int n,
                                         int num_classes) {
    if (!y_true || !y_pred || n < 0 || num_classes <= 0)
        return NULL;
    CMLConfusionMatrix* cm = cml_malloc(sizeof(CMLConfusionMatrix));
    if (!cm)
        return NULL;
    cm->num_classes = num_classes;
    cm->counts      = cml_calloc((size_t)num_classes * (size_t)num_classes, sizeof(int));
    if (!cm->counts) {
        cml_free(cm);
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        int t = y_true[i], p = y_pred[i];
        if (t >= 0 && t < num_classes && p >= 0 && p < num_classes)
            cm->counts[t * num_classes + p]++;
    }
    return cm;
}

/** Free a confusion matrix.
 * @param cm (CMLConfusionMatrix*)
 * @return (void)
 */
void cml_confusion_matrix_free(CMLConfusionMatrix* cm) {
    if (!cm)
        return;
    cml_free(cm->counts);
    cml_free(cm);
}

/** Fraction of exact matches between n true and predicted labels.
 * @param y_true (const int*)
 * @param y_pred (const int*)
 * @param n (int)
 * @return (float)
 */
float cml_accuracy(const int* y_true, const int* y_pred, int n) {
    if (!y_true || !y_pred || n <= 0)
        return 0.0f;
    int correct = 0;
    for (int i = 0; i < n; i++)
        correct += (y_true[i] == y_pred[i]);
    return (float)correct / (float)n;
}

/* TP/FP/FN for class c from the confusion matrix (row = true, col = pred). */
static void class_counts(const CMLConfusionMatrix* cm, int c, long* tp, long* fp, long* fn,
                         long* support) {
    int k  = cm->num_classes;
    long t = 0, f = 0, m = 0, s = 0;
    t = cm->counts[c * k + c];
    for (int i = 0; i < k; i++) {
        if (i != c)
            f += cm->counts[i * k + c]; /* predicted c but truly i */
        s += cm->counts[c * k + i];     /* truly c */
    }
    m   = s - t; /* truly c, predicted otherwise */
    *tp = t;
    *fp = f;
    *fn = m;
    if (support)
        *support = s;
}

/* Shared precision/recall/F1 core. kind: 0 precision, 1 recall, 2 f1. */
static float prf(const CMLConfusionMatrix* cm, CMLAverage average, float* per_class, int kind) {
    if (!cm)
        return 0.0f;
    int k = cm->num_classes;

    if (average == CML_AVG_MICRO) {
        long TP = 0, FP = 0, FN = 0;
        for (int c = 0; c < k; c++) {
            long tp, fp, fn;
            class_counts(cm, c, &tp, &fp, &fn, NULL);
            TP += tp;
            FP += fp;
            FN += fn;
        }
        float prec = (TP + FP) ? (float)TP / (float)(TP + FP) : 0.0f;
        float rec  = (TP + FN) ? (float)TP / (float)(TP + FN) : 0.0f;
        if (kind == 0)
            return prec;
        if (kind == 1)
            return rec;
        return (prec + rec) ? 2.0f * prec * rec / (prec + rec) : 0.0f;
    }

    double acc = 0.0, wsum = 0.0, total_support = 0.0;
    for (int c = 0; c < k; c++) {
        long tp, fp, fn, support;
        class_counts(cm, c, &tp, &fp, &fn, &support);
        float prec = (tp + fp) ? (float)tp / (float)(tp + fp) : 0.0f;
        float rec  = (tp + fn) ? (float)tp / (float)(tp + fn) : 0.0f;
        float v    = kind == 0   ? prec
                     : kind == 1 ? rec
                                 : ((prec + rec) ? 2.0f * prec * rec / (prec + rec) : 0.0f);
        if (per_class)
            per_class[c] = v;
        if (average == CML_AVG_WEIGHTED) {
            acc += (double)v * (double)support;
            total_support += (double)support;
        } else {
            acc += v; /* macro and per-class both average unweighted */
            wsum += 1.0;
        }
    }
    if (average == CML_AVG_WEIGHTED)
        return total_support > 0 ? (float)(acc / total_support) : 0.0f;
    return wsum > 0 ? (float)(acc / wsum) : 0.0f;
}

/** Precision averaged per @p average; @p per_class (optional) gets per-class values.
 * @param cm (const CMLConfusionMatrix*)
 * @param average (CMLAverage)
 * @param per_class (float*)
 * @return (float)
 */
float cml_precision(const CMLConfusionMatrix* cm, CMLAverage average, float* per_class) {
    return prf(cm, average, per_class, 0);
}

/** Recall averaged per @p average; @p per_class (optional) gets per-class values.
 * @param cm (const CMLConfusionMatrix*)
 * @param average (CMLAverage)
 * @param per_class (float*)
 * @return (float)
 */
float cml_recall(const CMLConfusionMatrix* cm, CMLAverage average, float* per_class) {
    return prf(cm, average, per_class, 1);
}

/** F1 averaged per @p average; @p per_class (optional) gets per-class values.
 * @param cm (const CMLConfusionMatrix*)
 * @param average (CMLAverage)
 * @param per_class (float*)
 * @return (float)
 */
float cml_f1_score(const CMLConfusionMatrix* cm, CMLAverage average, float* per_class) {
    return prf(cm, average, per_class, 2);
}

/* Ascending comparator on the scores pointed at by paired index entries. */
typedef struct {
    float score;
    int label;
} ScoreLabel;

static int scorelabel_cmp(const void* a, const void* b) {
    float x = ((const ScoreLabel*)a)->score, y = ((const ScoreLabel*)b)->score;
    return (x > y) - (x < y);
}

/** Binary ROC AUC via the rank-sum (Mann-Whitney U) identity, averaging tied ranks.
 * @param y_true (const int*)
 * @param y_score (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_roc_auc(const int* y_true, const float* y_score, int n) {
    if (!y_true || !y_score || n <= 0)
        return 0.5f;
    ScoreLabel* sl = cml_malloc((size_t)n * sizeof(ScoreLabel));
    if (!sl)
        return 0.5f;
    long n_pos = 0, n_neg = 0;
    for (int i = 0; i < n; i++) {
        sl[i].score = y_score[i];
        sl[i].label = y_true[i] ? 1 : 0;
        if (sl[i].label)
            n_pos++;
        else
            n_neg++;
    }
    if (n_pos == 0 || n_neg == 0) {
        cml_free(sl);
        return 0.5f;
    }
    qsort(sl, (size_t)n, sizeof(ScoreLabel), scorelabel_cmp);
    /* Assign average ranks (1-based), summing ranks of positive samples. */
    double rank_sum_pos = 0.0;
    int i               = 0;
    while (i < n) {
        int j = i;
        while (j < n && sl[j].score == sl[i].score)
            j++;
        double avg_rank = ((double)(i + 1) + (double)j) / 2.0; /* ranks i+1..j */
        for (int t = i; t < j; t++)
            if (sl[t].label)
                rank_sum_pos += avg_rank;
        i = j;
    }
    double auc = (rank_sum_pos - (double)n_pos * ((double)n_pos + 1.0) / 2.0) /
                 ((double)n_pos * (double)n_neg);
    cml_free(sl);
    return (float)auc;
}

/** Coefficient of determination R^2 = 1 - SS_res / SS_tot.
 * @param y_true (const float*)
 * @param y_pred (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_r2_score(const float* y_true, const float* y_pred, int n) {
    if (!y_true || !y_pred || n <= 0)
        return 0.0f;
    double mean = 0.0;
    for (int i = 0; i < n; i++)
        mean += y_true[i];
    mean /= n;
    double ss_res = 0.0, ss_tot = 0.0;
    for (int i = 0; i < n; i++) {
        double e = (double)y_true[i] - (double)y_pred[i];
        double d = (double)y_true[i] - mean;
        ss_res += e * e;
        ss_tot += d * d;
    }
    if (ss_tot == 0.0)
        return ss_res == 0.0 ? 1.0f : 0.0f;
    return (float)(1.0 - ss_res / ss_tot);
}

/** Root mean squared error.
 * @param y_true (const float*)
 * @param y_pred (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_rmse(const float* y_true, const float* y_pred, int n) {
    if (!y_true || !y_pred || n <= 0)
        return 0.0f;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double e = (double)y_true[i] - (double)y_pred[i];
        s += e * e;
    }
    return (float)sqrt(s / n);
}

/** Mean absolute error.
 * @param y_true (const float*)
 * @param y_pred (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_mae_metric(const float* y_true, const float* y_pred, int n) {
    if (!y_true || !y_pred || n <= 0)
        return 0.0f;
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += fabs((double)y_true[i] - (double)y_pred[i]);
    return (float)(s / n);
}
