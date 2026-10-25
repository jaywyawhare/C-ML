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

/** Top-k accuracy: fraction of samples whose true class is among the k highest logits.
 * @param logits (const float*)
 * @param n_samples (int)
 * @param n_classes (int)
 * @param y_true (const int*)
 * @param k (int)
 * @return (float)
 */
float cml_top_k_accuracy(const float* logits, int n_samples, int n_classes, const int* y_true,
                         int k) {
    if (!logits || !y_true || n_samples <= 0 || n_classes <= 0 || k <= 0)
        return 0.0f;
    int correct = 0;
    for (int s = 0; s < n_samples; s++) {
        int t = y_true[s];
        if (t < 0 || t >= n_classes)
            continue;
        const float* row = logits + (size_t)s * n_classes;
        float tv         = row[t];
        /* In top-k iff fewer than k classes score strictly higher than the true class. */
        int higher = 0;
        for (int c = 0; c < n_classes; c++)
            if (row[c] > tv)
                higher++;
        if (higher < k)
            correct++;
    }
    return (float)correct / (float)n_samples;
}

/** Perplexity exp(mean(nll)) from per-token negative log-likelihoods.
 * @param nll (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_perplexity(const float* nll, int n) {
    if (!nll || n <= 0)
        return 0.0f;
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += nll[i];
    return (float)exp(s / n);
}

/** Intersection-over-union of two axis-aligned boxes [x1, y1, x2, y2].
 * @param a (const float*)
 * @param b (const float*)
 * @return (float)
 */
float cml_iou(const float* a, const float* b) {
    if (!a || !b)
        return 0.0f;
    float ix1 = a[0] > b[0] ? a[0] : b[0];
    float iy1 = a[1] > b[1] ? a[1] : b[1];
    float ix2 = a[2] < b[2] ? a[2] : b[2];
    float iy2 = a[3] < b[3] ? a[3] : b[3];
    float iw = ix2 - ix1, ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f)
        return 0.0f;
    float inter  = iw * ih;
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    float uni    = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/** Average precision (area under the PR curve) for one class.
 * @param y_true (const int*)
 * @param y_score (const float*)
 * @param n (int)
 * @return (float)
 */
float cml_average_precision(const int* y_true, const float* y_score, int n) {
    if (!y_true || !y_score || n <= 0)
        return 0.0f;
    ScoreLabel* sl = cml_malloc((size_t)n * sizeof(ScoreLabel));
    if (!sl)
        return 0.0f;
    long total_pos = 0;
    for (int i = 0; i < n; i++) {
        sl[i].score = y_score[i];
        sl[i].label = y_true[i] ? 1 : 0;
        total_pos += sl[i].label;
    }
    if (total_pos == 0) {
        cml_free(sl);
        return 0.0f;
    }
    /* Sort descending by score (reuse the ascending comparator, iterate in reverse). */
    qsort(sl, (size_t)n, sizeof(ScoreLabel), scorelabel_cmp);
    long tp = 0, fp = 0;
    double ap = 0.0, prev_recall = 0.0;
    for (int i = n - 1; i >= 0; i--) {
        if (sl[i].label)
            tp++;
        else
            fp++;
        double recall    = (double)tp / (double)total_pos;
        double precision = (double)tp / (double)(tp + fp);
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
    }
    cml_free(sl);
    return (float)ap;
}

/* Clipped count of matching n-grams: for each candidate n-gram, consume one equal,
 * not-yet-consumed reference n-gram. Returns the number matched. */
static int clipped_ngram_matches(const int* cand, int cl, const int* ref, int rl, int n) {
    int cand_ng = cl - n + 1, ref_ng = rl - n + 1;
    if (cand_ng <= 0 || ref_ng <= 0)
        return 0;
    char* used = cml_calloc((size_t)ref_ng, 1);
    if (!used)
        return 0;
    int matched = 0;
    for (int i = 0; i < cand_ng; i++) {
        for (int j = 0; j < ref_ng; j++) {
            if (used[j])
                continue;
            int eq = 1;
            for (int t = 0; t < n; t++)
                if (cand[i + t] != ref[j + t]) {
                    eq = 0;
                    break;
                }
            if (eq) {
                used[j] = 1;
                matched++;
                break;
            }
        }
    }
    cml_free(used);
    return matched;
}

/** Sentence BLEU over token-id sequences (brevity penalty x geometric-mean n-gram precision).
 * @param candidate (const int*)
 * @param cand_len (int)
 * @param reference (const int*)
 * @param ref_len (int)
 * @param max_n (int)
 * @return (float)
 */
float cml_bleu(const int* candidate, int cand_len, const int* reference, int ref_len, int max_n) {
    if (!candidate || !reference || cand_len <= 0 || ref_len <= 0 || max_n <= 0)
        return 0.0f;
    double log_sum = 0.0;
    for (int n = 1; n <= max_n; n++) {
        int cand_ng = cand_len - n + 1;
        if (cand_ng <= 0)
            return 0.0f; /* candidate too short for this n */
        int m = clipped_ngram_matches(candidate, cand_len, reference, ref_len, n);
        if (m == 0)
            return 0.0f; /* zero precision collapses BLEU to 0 */
        log_sum += log((double)m / (double)cand_ng);
    }
    double geo = exp(log_sum / max_n);
    double bp  = cand_len >= ref_len ? 1.0 : exp(1.0 - (double)ref_len / (double)cand_len);
    return (float)(bp * geo);
}
