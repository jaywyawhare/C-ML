#ifndef CML_OPS_IR_TC_OPT_H
#define CML_OPS_IR_TC_OPT_H

#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_TC_DEFAULT_MIN_DIM 16

typedef struct CMLTCConfig {
    int min_m, min_n, min_k;
    /* Native tensor-core tile in each dimension; dims are padded up to these.
     * Populated per detected hardware (WMMA 16, AMX 32, XMX 8x16x8). */
    int tile_m, tile_n, tile_k;
    bool allow_padding;
    bool prefer_fp16;
} CMLTCConfig;

int cml_tc_optimize(CMLGraph_t graph);
bool cml_tc_available(void);
void cml_tc_set_config(CMLTCConfig* config);
CMLTCConfig cml_tc_get_config(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_TC_OPT_H */
