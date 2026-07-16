/*
 * experiment.h — lightweight W&B-style experiment tracking (prototype).
 *
 * Emits an append-only event log per run under .cml/experiments/runs/<id>/.
 * A separate server (viz/exp_server.py) ingests these into SQLite and serves
 * a comparison UI. This keeps the C side dependency-free (just stdio).
 *
 * Layers exercised here:
 *   L1  scalars          cml_exp_log_scalar
 *   L4  config/hparams    cml_exp_config_* (at init or later)
 *   L6  histograms        cml_exp_log_histogram (raw values binned in C)
 *   L7  system metrics    cml_exp_log_system (/proc sampling)
 *   L8  media / tables    cml_exp_log_image, cml_exp_log_table
 *   L10 artifacts         cml_exp_log_artifact
 * Multi-run/persistence (L2), comparison (L3), dynamic panels (L5) and
 * sweeps (L9) live in the server + UI; the C side just tags runs with a
 * sweep id + params via the config.
 */
#ifndef CML_EXPERIMENT_H
#define CML_EXPERIMENT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLRun CMLRun;

/* Start a run. project/name may be NULL (defaults applied). config_json is an
 * optional JSON object string of hyperparameters, e.g. "{\"lr\":0.01}".
 * Returns NULL on failure. Root dir defaults to ./.cml/experiments (override
 * with the CML_EXP_DIR env var). */
CMLRun* cml_exp_run_init(const char* project, const char* name, const char* config_json);

/* Tag a run as part of a sweep (shows up in parallel-coordinates). */
void cml_exp_set_sweep(CMLRun* run, const char* sweep_id);

/* L1: log a scalar metric at a training step. */
void cml_exp_log_scalar(CMLRun* run, const char* name, long step, double value);

/* L4: add/replace a config key (value_json is a raw JSON scalar/string). */
void cml_exp_config_set(CMLRun* run, const char* key, const char* value_json);

/* L6: log a histogram of raw values (binned into `bins` buckets in C). */
void cml_exp_log_histogram(CMLRun* run, const char* name, long step,
                           const float* values, size_t n, int bins);

/* L7: sample this process's CPU%/RSS from /proc and log them at `step`. */
void cml_exp_log_system(CMLRun* run, long step);

/* L8: log an RGB image (row-major, 3 bytes/pixel). Stored raw; the UI renders
 * it onto a canvas. Good for confusion matrices, weight visualizations, etc. */
void cml_exp_log_image(CMLRun* run, const char* name, long step,
                       const unsigned char* rgb, int width, int height);

/* L8b: log a small table as CSV text (first row = header). */
void cml_exp_log_table(CMLRun* run, const char* name, long step, const char* csv);

/* L10: record an artifact (model/dataset) by path; size + fnv hash captured.
 * aliases is an optional comma-separated list (e.g. "latest,best") for the
 * model registry; may be NULL. */
void cml_exp_log_artifact(CMLRun* run, const char* name, const char* type,
                          const char* path, const char* aliases);

/* Run management (W&B parity extras) --------------------------------------- */
/* Tag a run (repeatable); tags drive filtering/grouping in the UI. */
void cml_exp_add_tag(CMLRun* run, const char* tag);
/* Free-form markdown notes shown on the run's overview. */
void cml_exp_set_notes(CMLRun* run, const char* notes);
/* Capture a line of console output (stdout/stderr) for the Logs tab. */
void cml_exp_log_console(CMLRun* run, const char* line);
/* Set a custom summary value (the "final" number shown in the run table). */
void cml_exp_summary_set(CMLRun* run, const char* key, double value);

/* Raise an alert (level: "info"/"warn"/"error") — surfaces in the Alerts tab
 * and can flag diverging/NaN runs. */
void cml_exp_alert(CMLRun* run, const char* level, const char* message);

/* Log an x/y curve (PR curve, ROC, calibration, ...). Rendered as a line plot
 * in the Curves tab. xlabel/ylabel are axis titles (may be NULL). */
void cml_exp_log_curve(CMLRun* run, const char* name, const char* xlabel,
                       const char* ylabel, const float* xs, const float* ys, size_t n);

/* Mark the run finished ("finished"/"crashed"/...) and flush. Also records
 * end time + duration + captured summary. */
void cml_exp_run_finish(CMLRun* run, const char* status);

/* Accessor: the run's id (stable string). */
const char* cml_exp_run_id(const CMLRun* run);

#ifdef __cplusplus
}
#endif

#endif /* CML_EXPERIMENT_H */
