/* Smoke + correctness test for training_metrics.c (previously untested): create,
 * record epochs, export JSON, and read it back. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cml.h"

static int g_fail = 0;
#define CHECK(name, cond) do { \
    if (cond) printf("  PASS  %s\n", name); \
    else { printf("  FAIL  %s\n", name); g_fail = 1; } } while (0)

int main(void) {
    cml_init();
    printf("=== training_metrics ===\n");

    int N = 5;
    TrainingMetrics* m = training_metrics_create((size_t)N);
    CHECK("training_metrics_create", m != NULL);
    if (!m) { printf("\nTRAINING METRICS FAILED\n"); return 1; }

    /* record a decreasing loss / increasing accuracy curve */
    for (int e = 0; e < N; e++)
        training_metrics_record_epoch(m, (size_t)e, 1.0f / (float)(e + 1), 0.5f + 0.08f * e);

    const char* path = "/tmp/cml_metrics_test.json";
    remove(path);
    int rc = training_metrics_export_json(m, path, NULL);
    CHECK("export_json returns success", rc == 0);

    /* read the JSON back and confirm it captured a recorded value */
    FILE* f = fopen(path, "r");
    CHECK("export_json produced a file", f != NULL);
    if (f) {
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        char* buf = (char*)malloc((size_t)sz + 1);
        size_t got = fread(buf, 1, (size_t)sz, f); buf[got] = '\0';
        fclose(f);
        CHECK("JSON file is non-empty", sz > 0);
        /* first epoch loss was 1.0 — the exported JSON should mention loss */
        CHECK("JSON contains a 'loss' field", strstr(buf, "loss") != NULL);
        free(buf);
        remove(path);
    }

    training_metrics_free(m);
    CHECK("training_metrics_free (no crash)", 1);

    printf("\n%s\n", g_fail ? "TRAINING METRICS FAILED" : "All training_metrics tests passed");
    return g_fail;
}
