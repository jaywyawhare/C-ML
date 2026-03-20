#ifndef CML_OPS_IR_LINEARIZE_H
#define CML_OPS_IR_LINEARIZE_H

#include "ops/ir/schedule.h"
#include "ops/ir/ir.h"
#include "tensor/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_VIRTUAL_REGS 64

typedef enum {
    LINOP_LOAD,     /* Load tensor from memory into vreg */
    LINOP_COMPUTE,  /* Execute UOp, result in vreg */
    LINOP_STORE,    /* Store vreg to output tensor */
    LINOP_LOOP,     /* Loop header: axis extent in dest_reg, stride in src_regs[0] */
    LINOP_ENDLOOP,  /* Loop footer */
    LINOP_BARRIER,  /* Shared memory barrier */
    LINOP_LOCAL_ALLOC,  /* Allocate shared/local memory */
    LINOP_LOCAL_LOAD,   /* Load from shared memory */
    LINOP_LOCAL_STORE,  /* Store to shared memory */
} LinearOpKind;

typedef struct {
    LinearOpKind kind;
    UOpType uop;             /* The original operation (for COMPUTE) */
    int dest_reg;            /* Destination virtual register */
    int src_regs[8];         /* Source virtual registers */
    int num_srcs;
    Tensor* tensor;          /* Associated tensor (for LOAD/STORE) */
    bool is_eliminated;      /* True if this intermediate stays in reg */

    int loop_axis;           /* Axis index (for LOOP/ENDLOOP) */
    int loop_extent;         /* Trip count (for LOOP) */
    int loop_stride;         /* Stride (for LOOP) */
    int vec_width;           /* Vector width (for UPCAST-tagged ops) */
    size_t local_size;       /* Bytes (for LOCAL_ALLOC) */
} LinearOp;

typedef struct LinearProgram {
    LinearOp* ops;
    int num_ops;
    int capacity;
    int next_vreg;           /* Next free virtual register */

    int* loop_axes;          /* Axis extents for each loop dimension */
    int num_axes;
    int axes_capacity;

    bool has_local_memory;   /* Whether any local memory is in use */
    size_t local_mem_used;   /* Total shared memory allocated */
    int group_dims[3];       /* Workgroup dimensions (x,y,z) */
} LinearProgram;

LinearProgram* linear_program_create(void);
void linear_program_free(LinearProgram* prog);
int linear_program_emit(LinearProgram* prog, LinearOp op);
int alloc_vreg(LinearProgram* prog);
const char* linop_name(LinearOpKind k);
void linear_program_print(const LinearProgram* prog);

LinearProgram* linearize_group(const CMLFusionGroup* g);
void cml_linearize_group_print(const CMLFusionGroup* g);
int cml_linearize_group_count(const CMLFusionGroup* g);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_LINEARIZE_H */
