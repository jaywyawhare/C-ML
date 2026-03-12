/**
 * @file compiler_viz.c
 * @brief Compiler visualization/debugger implementation
 */

#include "ops/ir/compiler_viz.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _POSIX_C_SOURCE
#include <time.h>
static double viz_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#else
static double viz_get_time_ms(void) { return 0.0; }
#endif

CMLCompilerViz* cml_compiler_viz_create(const char* output_path) {
    CMLCompilerViz* viz = (CMLCompilerViz*)calloc(1, sizeof(CMLCompilerViz));
    if (!viz) return NULL;

    viz->enabled = true;
    if (output_path)
        strncpy(viz->output_path, output_path, sizeof(viz->output_path) - 1);

    return viz;
}

void cml_compiler_viz_free(CMLCompilerViz* viz) {
    if (!viz) return;
    CMLVizEvent* e = viz->events;
    while (e) {
        CMLVizEvent* next = e->next;
        free(e->ir_snapshot);
        free(e);
        e = next;
    }
    free(viz);
}

void cml_compiler_viz_enable(CMLCompilerViz* viz, bool enable) {
    if (viz) viz->enabled = enable;
}

int cml_compiler_viz_record(CMLCompilerViz* viz, CMLVizEventType type,
                             const char* description, CMLGraph_t ir) {
    if (!viz || !viz->enabled) return 0;

    CMLVizEvent* event = (CMLVizEvent*)calloc(1, sizeof(CMLVizEvent));
    if (!event) return -1;

    event->type = type;
    event->timestamp_ms = viz_get_time_ms();
    if (description)
        strncpy(event->description, description, sizeof(event->description) - 1);

    if (ir) {
        event->ir_snapshot = cml_ir_to_string(ir);
    }

    /* Append to list */
    if (viz->tail) {
        viz->tail->next = event;
    } else {
        viz->events = event;
    }
    viz->tail = event;
    viz->num_events++;

    return 0;
}

int cml_compiler_viz_export(CMLCompilerViz* viz) {
    if (!viz || viz->output_path[0] == '\0') return -1;

    FILE* f = fopen(viz->output_path, "w");
    if (!f) return -1;

    fprintf(f, "{\n  \"events\": [\n");

    CMLVizEvent* e = viz->events;
    int i = 0;
    while (e) {
        if (i > 0) fprintf(f, ",\n");

        const char* type_str;
        switch (e->type) {
        case CML_VIZ_IR_CREATED:            type_str = "ir_created"; break;
        case CML_VIZ_OPTIMIZATION_APPLIED:  type_str = "optimization"; break;
        case CML_VIZ_FUSION_DECISION:       type_str = "fusion"; break;
        case CML_VIZ_SCHEDULE_CREATED:      type_str = "schedule"; break;
        case CML_VIZ_KERNEL_GENERATED:      type_str = "kernel"; break;
        case CML_VIZ_MEMORY_PLANNED:        type_str = "memory"; break;
        case CML_VIZ_EXECUTION_STARTED:     type_str = "exec_start"; break;
        case CML_VIZ_EXECUTION_FINISHED:    type_str = "exec_end"; break;
        default:                            type_str = "unknown"; break;
        }

        fprintf(f, "    {\"type\": \"%s\", \"time_ms\": %.3f, \"desc\": \"%s\"}",
                type_str, e->timestamp_ms, e->description);

        e = e->next;
        i++;
    }

    fprintf(f, "\n  ],\n  \"total_events\": %d\n}\n", viz->num_events);
    fclose(f);
    return 0;
}

int cml_compiler_viz_num_events(const CMLCompilerViz* viz) {
    return viz ? viz->num_events : 0;
}

void cml_compiler_viz_clear(CMLCompilerViz* viz) {
    if (!viz) return;
    CMLVizEvent* e = viz->events;
    while (e) {
        CMLVizEvent* next = e->next;
        free(e->ir_snapshot);
        free(e);
        e = next;
    }
    viz->events = NULL;
    viz->tail = NULL;
    viz->num_events = 0;
}
