#pragma once
#include "ops/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    UOP_RANGE,
    UOP_INDEX,
} RangeUOpType;

typedef struct {
    int start;
    int end;
    int step;
    int dim;
} RangeInfo;

typedef struct {
    int num_ranges;
    int range_ids[8];
    size_t strides[8];
} IndexInfo;

typedef struct RangeNode {
    RangeUOpType type;
    union {
        RangeInfo range;
        IndexInfo index;
    };
    int id;
    struct RangeNode* next;
} RangeNode;

typedef struct RangeProgram {
    RangeNode* head;
    RangeNode* tail;
    int num_nodes;
    int next_id;
} RangeProgram;

RangeProgram* range_program_create(void);
void range_program_free(RangeProgram* prog);
int range_program_add_range(RangeProgram* prog, int start, int end, int step, int dim);
int range_program_add_index(RangeProgram* prog, int* range_ids, size_t* strides, int num_ranges);
void range_program_print(const RangeProgram* prog);

int cml_rangeify(CMLGraph_t graph);

#ifdef __cplusplus
}
#endif
