#ifndef CML_DISTRIBUTED_RING_ALLREDUCE_H
#define CML_DISTRIBUTED_RING_ALLREDUCE_H

#include "distributed/distributed.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cml_ring_allreduce(float* data, size_t count, int world_size, int rank,
                       DistReduceOp op, DistCommOps* ops, void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* CML_DISTRIBUTED_RING_ALLREDUCE_H */
