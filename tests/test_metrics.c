/* Evaluation metrics verified against hand-computed values: confusion matrix,
 * accuracy, precision/recall/F1 (macro/micro/weighted/per-class), binary ROC-AUC
 * (incl. ties), and regression R^2/RMSE/MAE. */
#include <math.h>
#include <stdio.h>
#include "core/metrics.h"
#include "test_harness.h"

#define NEAR(a, b) (fabsf((float)(a) - (float)(b)) < 1e-4f)

static int test_confusion_and_accuracy(void) {
    /* 3-class, 6 samples. */
    int yt[6]              = {0, 0, 1, 1, 2, 2};
    int yp[6]              = {0, 1, 1, 1, 2, 0};
    CMLConfusionMatrix* cm = cml_confusion_matrix(yt, yp, 6, 3);
    if (!cm)
        return 0;
    /* row=true,col=pred: [ [1,1,0],[0,2,0],[1,0,1] ] */
    int ok = cm->counts[0] == 1 && cm->counts[1] == 1 && cm->counts[2] == 0 && cm->counts[3] == 0 &&
             cm->counts[4] == 2 && cm->counts[5] == 0 && cm->counts[6] == 1 && cm->counts[7] == 0 &&
             cm->counts[8] == 1;
    ok &= NEAR(cml_accuracy(yt, yp, 6), 4.0f / 6.0f); /* diag 1+2+1=4 */
    cml_confusion_matrix_free(cm);
    return ok;
}

static int test_precision_recall_f1(void) {
    int yt[6]              = {0, 0, 1, 1, 2, 2};
    int yp[6]              = {0, 1, 1, 1, 2, 0};
    CMLConfusionMatrix* cm = cml_confusion_matrix(yt, yp, 6, 3);
    float pc[3];
    /* Precision per class: c0 TP1/(1+1)=0.5, c1 TP2/(2+1)=0.667, c2 TP1/(1+0)=1.0 */
    float prec_macro = cml_precision(cm, CML_AVG_MACRO, pc);
    int ok           = NEAR(pc[0], 0.5f) && NEAR(pc[1], 2.0f / 3.0f) && NEAR(pc[2], 1.0f);
    ok &= NEAR(prec_macro, (0.5f + 2.0f / 3.0f + 1.0f) / 3.0f);
    /* Recall per class: c0 1/2=0.5, c1 2/2=1.0, c2 1/2=0.5 */
    float rec_macro = cml_recall(cm, CML_AVG_MACRO, pc);
    ok &= NEAR(pc[0], 0.5f) && NEAR(pc[1], 1.0f) && NEAR(pc[2], 0.5f);
    ok &= NEAR(rec_macro, (0.5f + 1.0f + 0.5f) / 3.0f);
    /* Micro P=R=F1=accuracy for single-label = 4/6. */
    ok &= NEAR(cml_precision(cm, CML_AVG_MICRO, NULL), 4.0f / 6.0f);
    ok &= NEAR(cml_f1_score(cm, CML_AVG_MICRO, NULL), 4.0f / 6.0f);
    /* F1 c0 = 2*.5*.5/(1)=0.5; weighted by support (2,2,2 -> equal) = macro. */
    float f1_macro = cml_f1_score(cm, CML_AVG_MACRO, pc);
    ok &= NEAR(pc[0], 0.5f);
    ok &= NEAR(cml_f1_score(cm, CML_AVG_WEIGHTED, NULL), f1_macro); /* equal supports */
    cml_confusion_matrix_free(cm);
    return ok;
}

static int test_roc_auc(void) {
    /* Perfect separation -> AUC 1.0 */
    int yt1[4]   = {0, 0, 1, 1};
    float ys1[4] = {0.1f, 0.2f, 0.8f, 0.9f};
    int ok       = NEAR(cml_roc_auc(yt1, ys1, 4), 1.0f);
    /* Reversed -> 0.0 */
    float ys2[4] = {0.9f, 0.8f, 0.2f, 0.1f};
    ok &= NEAR(cml_roc_auc(yt1, ys2, 4), 0.0f);
    /* One pos, one neg, tied score -> 0.5 (averaged rank) */
    int yt3[2]   = {0, 1};
    float ys3[2] = {0.5f, 0.5f};
    ok &= NEAR(cml_roc_auc(yt3, ys3, 2), 0.5f);
    /* Known mixed case: scores 0.1,0.4,0.35,0.8 labels 0,0,1,1 -> AUC 0.75 */
    int yt4[4]   = {0, 0, 1, 1};
    float ys4[4] = {0.1f, 0.4f, 0.35f, 0.8f};
    ok &= NEAR(cml_roc_auc(yt4, ys4, 4), 0.75f);
    return ok;
}

static int test_regression(void) {
    float yt[4] = {3.0f, -0.5f, 2.0f, 7.0f};
    float yp[4] = {2.5f, 0.0f, 2.0f, 8.0f};
    /* MAE = (0.5+0.5+0+1)/4 = 0.5; RMSE = sqrt((0.25+0.25+0+1)/4)=sqrt(0.375) */
    int ok = NEAR(cml_mae_metric(yt, yp, 4), 0.5f);
    ok &= NEAR(cml_rmse(yt, yp, 4), sqrtf(0.375f));
    /* R^2: mean=2.875; SS_tot=(0.0156+11.39+0.766+16.9)=29.19; SS_res=1.5 -> ~0.9486 */
    ok &= NEAR(cml_r2_score(yt, yp, 4), 0.94860814f);
    /* Perfect prediction -> R^2 = 1 */
    ok &= NEAR(cml_r2_score(yt, yt, 4), 1.0f);
    return ok;
}

int main(void) {
    printf("=== evaluation metrics ===\n");
    TEST(confusion_and_accuracy);
    TEST(precision_recall_f1);
    TEST(roc_auc);
    TEST(regression);
    return TEST_SUMMARY();
}
