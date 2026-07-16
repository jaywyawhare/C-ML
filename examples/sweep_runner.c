/*
 * sweep_runner.c — a real hyperparameter *sweep controller* (grid search).
 *
 * Unlike experiment_demo (which logs a fixed 3-run sweep), this orchestrates a
 * full grid over {lr} x {batch_size} x {optimizer}, launches a run per cell,
 * fires alerts on unstable configs, and reports the best cell — i.e. the "agent"
 * half of W&B Sweeps, not just the visualization.
 */
#include "core/experiment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

static float frand(unsigned* s) { *s = *s * 1103515245u + 12345u; return ((*s >> 16) & 0x7fff) / 32768.0f; }
static float gauss(unsigned* s) { float u1 = frand(s) + 1e-7f, u2 = frand(s); return sqrtf(-2 * logf(u1)) * cosf(6.2831853f * u2); }

int main(void) {
    const char* sweep_id = "grid-sweep-2";
    float  lrs[] = {0.001f, 0.01f, 0.05f, 0.1f};
    int    bss[] = {16, 64};
    const char* opts[] = {"adam", "sgd"};
    int nl = 4, nb = 2, no = 2, epochs = 50;

    float best_acc = -1.0f; char best_name[64] = "";
    int cell = 0, total = nl * nb * no;

    for (int li = 0; li < nl; li++)
      for (int bi = 0; bi < nb; bi++)
        for (int oi = 0; oi < no; oi++) {
            cell++;
            float lr = lrs[li]; int bs = bss[bi]; const char* opt = opts[oi];
            char cfg[192];
            snprintf(cfg, sizeof(cfg), "{\"lr\":%.4f,\"batch_size\":%d,\"optimizer\":\"%s\"}", lr, bs, opt);
            char name[80];
            snprintf(name, sizeof(name), "%s-lr%.3g-bs%d", opt, lr, bs);

            CMLRun* run = cml_exp_run_init("iris-sweep", name, cfg);
            cml_exp_set_sweep(run, sweep_id);
            cml_exp_add_tag(run, opt);
            cml_exp_add_tag(run, "grid");
            printf("[%2d/%d] launching %s ... ", cell, total, name); fflush(stdout);

            unsigned seed = 4242u + (unsigned)cell * 131u;
            /* adam converges faster; sgd slower; high lr unstable */
            float speed = (strcmp(opt, "adam") == 0 ? 1.0f : 0.6f) * (bs == 16 ? 1.1f : 0.9f);
            float decay = speed * (lr >= 0.1f ? 0.10f : lr >= 0.05f ? 0.08f : lr >= 0.01f ? 0.06f : 0.025f);
            int unstable = (lr >= 0.1f);
            float best = 0.0f, final_loss = 1.0f;

            int W = 128; float* w = malloc(sizeof(float) * W);
            for (int i = 0; i < W; i++) w[i] = gauss(&seed) * 0.5f;

            for (int e = 1; e <= epochs; e++) {
                float base = 0.9f * expf(-decay * e);
                float noise = unstable ? 0.10f : 0.015f;
                float loss = base + noise * (frand(&seed) - 0.5f);
                if (unstable && frand(&seed) > 0.9f) loss += 0.3f; /* spikes */
                if (loss < 0.004f) loss = 0.004f + 0.002f * frand(&seed);
                float acc = 1.0f - loss * 0.9f - (unstable ? 0.06f * frand(&seed) : 0.0f);
                if (acc < 0) acc = 0; if (acc > 1) acc = 1;
                if (acc > best) best = acc;
                final_loss = loss;
                cml_exp_log_scalar(run, "train/loss", e, loss);
                cml_exp_log_scalar(run, "train/accuracy", e, acc);
                cml_exp_log_scalar(run, "val/accuracy", e, acc * (0.9f + 0.05f * frand(&seed)));
                cml_exp_log_scalar(run, "grad_norm", e, base * 3.0f + (unstable ? 2.0f * frand(&seed) : 0.1f));
                for (int i = 0; i < W; i++) w[i] -= lr * gauss(&seed) * base;
                if (e % 10 == 0) { cml_exp_log_histogram(run, "weights/layer0", e, w, W, 32); cml_exp_log_system(run, e); }
            }

            if (unstable) cml_exp_alert(run, "warn", "High learning rate — unstable/spiky training detected");
            if (final_loss > 0.3f) cml_exp_alert(run, "error", "Run did not converge (final loss > 0.3)");

            cml_exp_summary_set(run, "best_val_accuracy", best);
            cml_exp_summary_set(run, "final_loss", final_loss);
            char bstr[32]; snprintf(bstr, sizeof(bstr), "%.4f", best);
            cml_exp_config_set(run, "best_val_accuracy", bstr);

            int is_best = best > best_acc;
            cml_exp_log_artifact(run, "iris-sweep", "model", "/dev/null", is_best ? "best,latest" : "latest");
            cml_exp_run_finish(run, "finished");
            if (is_best) { best_acc = best; snprintf(best_name, sizeof(best_name), "%s", name); }
            printf("acc=%.3f%s\n", best, is_best ? "  <-- best" : "");
            free(w);
        }

    printf("\nSweep '%s' done: %d runs. Best = %s (acc %.3f)\n", sweep_id, total, best_name, best_acc);
    printf("View: python3 viz/serve.py  ->  http://localhost:6969\n");
    return 0;
}
