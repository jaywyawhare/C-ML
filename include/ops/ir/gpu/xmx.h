#ifndef CML_OPS_IR_GPU_XMX_H
#define CML_OPS_IR_GPU_XMX_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool cml_xmx_available(void);

typedef struct CMLXMXConfig {
    int dpas_depth;
    int exec_size;
    int ops_per_chan;
} CMLXMXConfig;

CMLXMXConfig cml_xmx_get_config(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_GPU_XMX_H */
