/*
 * hyperband_sweep.c — random search + successive-halving (Hyperband-lite).
 *
 * Samples N random hyperparameter configs, then runs them through rungs of
 * increasing budget, keeping only the top half each round and *early-stopping*
 * the rest (status "stopped"). Survivors log PR/ROC curves. This demonstrates
 * a real search strategy + early termination, beyond the grid in sweep_runner.
 */
#include "core/experiment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned lcg(unsigned* s) { *s = *s * 1103515245u + 12345u; return (*s >> 16) & 0x7fff; }
static float rnd(unsigned* s) { return lcg(s) / 32768.0f; }

typedef struct {
    CMLRun* run; float lr, dropout; int hidden;
    float ceiling, speed, acc; int alive, trained;
} Trial;

int main(void) {
    const char* sweep = "hyperband-3";
    enum { N = 8 };
    Trial t[N];
    unsigned seed = 99u;

    for (int i = 0; i < N; i++) {
        float lr = powf(10.0f, -3.0f + 2.0f * rnd(&seed));   /* 1e-3 .. 1e-1 */
        float dropout = 0.5f * rnd(&seed);
        int hidden = 16 + (int)(48 * rnd(&seed));
        char cfg[192];
        snprintf(cfg, sizeof(cfg), "{\"lr\":%.4f,\"dropout\":%.3f,\"hidden\":%d}", lr, dropout, hidden);
        char name[64];
        snprintf(name, sizeof(name), "trial-%d", i);
        CMLRun* run = cml_exp_run_init("hyperband-search", name, cfg);
        cml_exp_set_sweep(run, sweep);
        cml_exp_add_tag(run, "random");
        /* simulated quality: closeness to an ideal (lr~0.01, dropout~0.2, hidden~48) */
        float dl = fabsf(log10f(lr) - log10f(0.01f)) / 2.0f;
        float dd = fabsf(dropout - 0.2f) / 0.5f;
        float dh = fabsf((float)hidden - 48.0f) / 48.0f;
        float ceiling = 0.99f - 0.35f * (0.6f * dl + 0.2f * dd + 0.2f * dh);
        t[i] = (Trial){ run, lr, dropout, hidden, ceiling, 0.05f + 0.05f * rnd(&seed), 0.0f, 1, 0 };
    }

    int rungs[4] = {8, 16, 32, 50};
    int keep[4]  = {8, 4, 2, 1};

    for (int r = 0; r < 4; r++) {
        int budget = rungs[r];
        for (int i = 0; i < N; i++) {
            if (!t[i].alive) continue;
            for (int e = t[i].trained + 1; e <= budget; e++) {
                float acc = t[i].ceiling * (1.0f - expf(-t[i].speed * e)) + 0.01f * (rnd(&seed) - 0.5f);
                if (acc < 0) acc = 0;
                if (acc > 1) acc = 1;
                cml_exp_log_scalar(t[i].run, "train/loss", e, (1.0f - acc) * 0.9f + 0.02f);
                cml_exp_log_scalar(t[i].run, "val/accuracy", e, acc);
                t[i].acc = acc;
            }
            t[i].trained = budget;
        }
        /* successive halving: stop the worst until only keep[r+1] survive */
        if (r < 3) {
            int target = keep[r + 1];
            for (;;) {
                int alive = 0; for (int i = 0; i < N; i++) alive += t[i].alive;
                if (alive <= target) break;
                int worst = -1; float wa = 1e9f;
                for (int i = 0; i < N; i++) if (t[i].alive && t[i].acc < wa) { wa = t[i].acc; worst = i; }
                t[worst].alive = 0;
                cml_exp_alert(t[worst].run, "info", "Early-stopped by successive halving");
                cml_exp_summary_set(t[worst].run, "best_val_accuracy", t[worst].acc);
                char b[32]; snprintf(b, sizeof(b), "%.4f", t[worst].acc);
                cml_exp_config_set(t[worst].run, "best_val_accuracy", b);
                cml_exp_run_finish(t[worst].run, "stopped");
                printf("  stopped trial (acc %.3f) at rung budget %d\n", wa, budget);
            }
        }
    }

    for (int i = 0; i < N; i++) {
        if (!t[i].alive) continue;
        float px[11], py[11], rx[11], ry[11];
        for (int k = 0; k <= 10; k++) {
            float u = k / 10.0f;
            px[k] = u; py[k] = t[i].ceiling * (1.0f - 0.5f * u);            /* PR */
            rx[k] = u; ry[k] = 1.0f - (1.0f - t[i].ceiling) * (1.0f - u);   /* ROC */
        }
        cml_exp_log_curve(t[i].run, "pr_curve", "recall", "precision", px, py, 11);
        cml_exp_log_curve(t[i].run, "roc_curve", "fpr", "tpr", rx, ry, 11);
        cml_exp_summary_set(t[i].run, "best_val_accuracy", t[i].acc);
        char b[32]; snprintf(b, sizeof(b), "%.4f", t[i].acc);
        cml_exp_config_set(t[i].run, "best_val_accuracy", b);
        cml_exp_run_finish(t[i].run, "finished");
        printf("  survivor trial-%d acc %.3f (lr=%.4g dropout=%.2f hidden=%d)\n",
               i, t[i].acc, t[i].lr, t[i].dropout, t[i].hidden);
    }
    printf("Hyperband search complete: %d trials, halving 8->4->2->1.\n", N);
    return 0;
}
