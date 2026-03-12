/**
 * @file compiler_viz.h
 * @brief Hardware-level compiler visualization/debugger
 *
 * Visualizes IR transformations, kernel fusion decisions, register allocation,
 * and memory access patterns. Outputs to JSON for web UI consumption.
 */

#ifndef CML_COMPILER_VIZ_H
#define CML_COMPILER_VIZ_H

#include "ops/ir/ir.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Visualization event type */
typedef enum {
    CML_VIZ_IR_CREATED = 0,
    CML_VIZ_OPTIMIZATION_APPLIED,
    CML_VIZ_FUSION_DECISION,
    CML_VIZ_SCHEDULE_CREATED,
    CML_VIZ_KERNEL_GENERATED,
    CML_VIZ_MEMORY_PLANNED,
    CML_VIZ_EXECUTION_STARTED,
    CML_VIZ_EXECUTION_FINISHED,
} CMLVizEventType;

/** Visualization event */
typedef struct CMLVizEvent {
    CMLVizEventType type;
    double timestamp_ms;
    char description[256];
    char* ir_snapshot;        /* JSON representation of IR at this point */
    struct CMLVizEvent* next;
} CMLVizEvent;

/** Compiler visualizer */
typedef struct CMLCompilerViz {
    bool enabled;
    CMLVizEvent* events;      /* Linked list of events */
    CMLVizEvent* tail;
    int num_events;
    char output_path[256];    /* Path for JSON output */
} CMLCompilerViz;

/** Create compiler visualizer */
CMLCompilerViz* cml_compiler_viz_create(const char* output_path);

/** Free compiler visualizer */
void cml_compiler_viz_free(CMLCompilerViz* viz);

/** Enable/disable visualization */
void cml_compiler_viz_enable(CMLCompilerViz* viz, bool enable);

/** Record an event */
int cml_compiler_viz_record(CMLCompilerViz* viz, CMLVizEventType type,
                             const char* description, CMLGraph_t ir);

/** Export all events to JSON file */
int cml_compiler_viz_export(CMLCompilerViz* viz);

/** Get number of recorded events */
int cml_compiler_viz_num_events(const CMLCompilerViz* viz);

/** Clear all recorded events */
void cml_compiler_viz_clear(CMLCompilerViz* viz);

#ifdef __cplusplus
}
#endif

#endif /* CML_COMPILER_VIZ_H */
