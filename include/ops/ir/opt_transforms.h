#ifndef CML_OPS_IR_OPT_TRANSFORMS_H
#define CML_OPS_IR_OPT_TRANSFORMS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OPT_LOCAL,    /* Split axis into local + group (shared memory on GPU) */
    OPT_GROUP,    /* Assign axis to workgroup dimension */
    OPT_UNROLL,   /* Unroll loop by factor */
    OPT_UPCAST,   /* Vectorize axis (vector types in codegen) */
    OPT_PADTO,    /* Pad axis to multiple of N (alignment) */
    OPT_NOLOCALS, /* Disable local memory usage */
} CMLOptType;

typedef struct CMLOpt {
    CMLOptType type;
    int axis;
    int amount;
} CMLOpt;

typedef struct CMLOptList {
    CMLOpt* opts;
    int num_opts;
    int capacity;
} CMLOptList;

struct LinearProgram;

CMLOptList* cml_opt_list_create(void);
void cml_opt_list_free(CMLOptList* list);
void cml_opt_list_add(CMLOptList* list, CMLOptType type, int axis, int amount);

int cml_opt_apply(CMLOptList* opts, struct LinearProgram* prog);

int cml_opt_enumerate(struct LinearProgram* prog, CMLOptList*** out_lists,
                      int* out_count, int max_combinations);

const char* cml_opt_type_name(CMLOptType type);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_OPT_TRANSFORMS_H */
