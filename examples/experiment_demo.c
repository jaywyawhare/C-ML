/*
 * experiment_demo.c — exercises all 10 tracking layers via the cml_exp_* API.
 *
 * Runs a 3-point learning-rate *sweep* (L9). Each run logs:
 *   L1 scalars (train/val loss+acc, grad_norm, lr)   L4 config/hparams
 *   L6 weight & gradient histograms                  L7 system metrics
 *   L8 a confusion-matrix image + a per-class table  L10 a model artifact
 * L2/L3/L5 (persistence, comparison, dynamic panels) are provided by the
 * server + UI reading the emitted runs.
 *
 * Training curves are simulated (this demo proves the tracking pipeline, not
 * the trainer) but every byte flows through the real C logging API.
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
    const char* sweep_id = "lr-sweep-1";
    float lrs[3] = {0.001f, 0.01f, 0.1f};
    int epochs = 60;

    for (int r = 0; r < 3; r++) {
        float lr = lrs[r];
        char cfg[160];
        snprintf(cfg, sizeof(cfg), "{\"lr\":%.4f,\"batch_size\":32,\"optimizer\":\"adam\"}", lr);
        char name[64];
        snprintf(name, sizeof(name), "mlp-lr%.4g", lr);

        CMLRun* run = cml_exp_run_init("iris-mlp", name, cfg);
        if (!run) { fprintf(stderr, "run_init failed\n"); return 1; }
        cml_exp_set_sweep(run, sweep_id);
        cml_exp_config_set(run, "epochs", "60");
        cml_exp_config_set(run, "model", "\"mlp-16-8-3\"");
        cml_exp_add_tag(run, "iris");
        cml_exp_add_tag(run, "mlp");
        cml_exp_add_tag(run, lr >= 0.1f ? "aggressive" : lr <= 0.001f ? "conservative" : "baseline");
        cml_exp_set_notes(run, "MLP-16-8-3 on Iris. LR sweep to find best generalization.\n"
                               "Adam optimizer, batch size 32, 60 epochs.");
        cml_exp_log_console(run, "[info] initialized mlp-16-8-3, optimizer=adam");

        unsigned seed = 12345u + (unsigned)r * 777u;
        float decay = (lr <= 0.001f) ? 0.03f : (lr >= 0.1f) ? 0.08f : 0.06f;
        float noise = (lr >= 0.1f) ? 0.06f : 0.01f;
        float best_acc = 0.0f;

        int W = 8 * 16; /* first-layer weight count */
        float* w = malloc(sizeof(float) * W);
        for (int i = 0; i < W; i++) w[i] = gauss(&seed) * 0.5f;

        for (int e = 1; e <= epochs; e++) {
            float base = 0.9f * expf(-decay * e);
            float loss = base + noise * (frand(&seed) - 0.5f);
            if (loss < 0.005f) loss = 0.005f + 0.002f * frand(&seed);
            float acc = 1.0f - loss * 0.9f - (lr >= 0.1f ? 0.05f * frand(&seed) : 0.0f);
            if (acc < 0) acc = 0;
                if (acc > 1) acc = 1;
            if (acc > best_acc) best_acc = acc;
            float val_loss = loss * (1.1f + 0.2f * frand(&seed));
            float val_acc = acc * (0.9f + 0.05f * frand(&seed));

            cml_exp_log_scalar(run, "train/loss", e, loss);
            cml_exp_log_scalar(run, "train/accuracy", e, acc);
            cml_exp_log_scalar(run, "val/loss", e, val_loss);
            cml_exp_log_scalar(run, "val/accuracy", e, val_acc);
            cml_exp_log_scalar(run, "grad_norm", e, base * 3.0f + 0.1f);
            cml_exp_log_scalar(run, "lr", e, lr);

            for (int i = 0; i < W; i++) w[i] -= lr * gauss(&seed) * base;
            if (e % 5 == 0) {
                cml_exp_log_histogram(run, "weights/layer0", e, w, (size_t)W, 32);
                float* g = malloc(sizeof(float) * W);
                for (int i = 0; i < W; i++) g[i] = gauss(&seed) * base;
                cml_exp_log_histogram(run, "grads/layer0", e, g, (size_t)W, 32);
                free(g);
            }
            if (e % 10 == 0) cml_exp_log_system(run, e);
            if (e % 20 == 0) {
                char cl[128];
                snprintf(cl, sizeof(cl), "[epoch %2d] loss=%.4f acc=%.3f val_acc=%.3f", e, loss, acc, val_acc);
                cml_exp_log_console(run, cl);
            }
            usleep(500);
        }

        /* L8: confusion-matrix image (3x3 scaled to 90x90 RGB). */
        int CS = 3, PX = 30, IW = CS * PX, IH = CS * PX;
        unsigned char* img = malloc((size_t)IW * IH * 3);
        float cm[3][3] = {{0.90f, 0.05f, 0.05f}, {0.10f, 0.85f, 0.05f}, {0.05f, 0.10f, 0.85f}};
        for (int y = 0; y < IH; y++)
            for (int x = 0; x < IW; x++) {
                float v = cm[y / PX][x / PX];
                unsigned char* p = &img[((size_t)y * IW + x) * 3];
                p[0] = (unsigned char)(30 + (1 - v) * 40);
                p[1] = (unsigned char)(v * 200 + 30);
                p[2] = (unsigned char)(80 + v * 100);
            }
        cml_exp_log_image(run, "confusion_matrix", epochs, img, IW, IH);
        free(img);

        /* L8b: per-class metric table. */
        cml_exp_log_table(run, "per_class", epochs,
                          "class,precision,recall,f1\n"
                          "setosa,0.98,0.99,0.985\n"
                          "versicolor,0.88,0.90,0.89\n"
                          "virginica,0.91,0.87,0.89");

        /* L10: write + log a model artifact. */
        char mp[256];
        snprintf(mp, sizeof(mp), ".cml/experiments/model_%d.bin", r);
        FILE* mf = fopen(mp, "wb");
        if (mf) { fwrite(w, sizeof(float), (size_t)W, mf); fclose(mf); }
        /* Model-registry aliases: the mid-LR run is our "best"; all get "latest". */
        cml_exp_log_artifact(run, "iris-mlp", "model", mp, (r == 1) ? "best,latest" : "latest");

        char summ[32];
        snprintf(summ, sizeof(summ), "%.4f", best_acc);
        cml_exp_config_set(run, "best_val_accuracy", summ);
        cml_exp_summary_set(run, "best_val_accuracy", best_acc);
        cml_exp_summary_set(run, "epochs_run", epochs);
        cml_exp_log_console(run, "[info] training complete, model saved");
        cml_exp_run_finish(run, "finished");
        free(w);
        printf("run %d (lr=%.4g) done — best_acc=%.3f  id=%s\n", r, lr, best_acc, "ok");
    }

    printf("\nSweep complete. Data in .cml/experiments/runs/\n");
    printf("Start the tracker UI:  python3 viz/serve.py   ->  http://localhost:6969\n");
    return 0;
}
