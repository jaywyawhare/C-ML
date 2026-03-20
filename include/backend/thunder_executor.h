#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLThunderExecutor {
    char backend_name[32];
    bool initialized;
    void* dispatch_ctx;
} CMLThunderExecutor;

CMLThunderExecutor* cml_thunder_create(const char* backend);
void cml_thunder_free(CMLThunderExecutor* exec);

typedef struct CMLThunderOp {
    const char* op_name;
    void** inputs;
    int num_inputs;
    void** outputs;
    int num_outputs;
} CMLThunderOp;

int cml_thunder_execute(CMLThunderExecutor* exec, CMLThunderOp* ops, int num_ops);
int cml_thunder_register(void);

#ifdef __cplusplus
}
#endif
