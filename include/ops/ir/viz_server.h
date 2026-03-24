

#ifndef CML_OPS_IR_VIZ_SERVER_H
#define CML_OPS_IR_VIZ_SERVER_H

#include "ops/ir/ir.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VizServer VizServer;

VizServer* viz_server_start(CMLGraph_t ir, int port);

void viz_server_update(VizServer* srv, CMLGraph_t ir);

int viz_server_poll(VizServer* srv);

void viz_server_stop(VizServer* srv);

int viz_server_port(const VizServer* srv);

char* viz_graph_to_json(CMLGraph_t ir);

int viz_export_html(CMLGraph_t ir, const char* path);

#ifdef __cplusplus
}
#endif

#endif 
