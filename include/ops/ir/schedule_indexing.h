

#ifndef CML_OPS_IR_SCHEDULE_INDEXING_H
#define CML_OPS_IR_SCHEDULE_INDEXING_H

#include "symbolic/symbolic.h"
#include "tensor/shape_tracker.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IndexMap {
    SymExpr* flat_index;
    SymExpr* valid;     
    int      num_vars;  
} IndexMap;

IndexMap* index_map_create(SymExpr* flat_index, SymExpr* valid, int num_vars);
void      index_map_free(IndexMap* im);
IndexMap* index_map_copy(const IndexMap* im);

typedef struct LoopVar {
    SymExpr* expr;      
    int64_t  begin;
    int64_t  end;
} LoopVar;

LoopVar* loop_vars_create(const int* shape, int n);
void     loop_vars_free(LoopVar* vars, int n);

IndexMap* schedule_build_index_map(const ShapeTracker* st,
                                   const LoopVar* loop_vars,
                                   int num_vars);

IndexMap* schedule_build_index_map_simplified(const ShapeTracker* st,
                                              const LoopVar* loop_vars,
                                              int num_vars);

int index_map_to_c(const IndexMap* im,
                   const char* const* var_names, int num_vars,
                   char* index_buf, size_t index_buf_size,
                   char* valid_buf, size_t valid_buf_size);

IndexMap* index_map_compose(const IndexMap* outer, const IndexMap* inner);

void index_map_print(const IndexMap* im);

#ifdef __cplusplus
}
#endif

#endif 
