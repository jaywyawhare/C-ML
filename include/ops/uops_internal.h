#ifndef CML_OPS_UOPS_INTERNAL_H
#define CML_OPS_UOPS_INTERNAL_H

/*
 * Shared helpers for the uop builders, split out of uops.c so cohesive builder
 * groups (optimizer steps, conv, etc.) can live in their own translation units
 * without duplicating the node-finishing boilerplate. Not part of the public API.
 */

#include "ops/ir/ir.h"     /* CMLGraph_t */
#include "tensor/tensor.h" /* Tensor, DType, DeviceType */

#ifdef __cplusplus
extern "C" {
#endif

/* Finalize the graph's tail node as a fresh source tensor of the given shape. */
Tensor* cml_uop_finish_source_node(CMLGraph_t ir, const int* shape, int ndim, DType dtype,
                                   DeviceType device);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_UOPS_INTERNAL_H */
