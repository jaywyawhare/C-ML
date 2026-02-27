/**
 * @file ring_allreduce.h
 * @brief Ring all-reduce algorithm for distributed training
 */

#ifndef CML_DISTRIBUTED_RING_ALLREDUCE_H
#define CML_DISTRIBUTED_RING_ALLREDUCE_H

#include "distributed/distributed.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Perform ring all-reduce on float data
 *
 * Two-phase algorithm:
 *   Phase 1: Reduce-scatter ring — each rank ends up with a reduced chunk
 *   Phase 2: All-gather ring — each rank collects all reduced chunks
 *
 * @param data     Float buffer (modified in-place)
 * @param count    Number of elements
 * @param world_size Total ranks
 * @param rank     This rank
 * @param op       Reduction operation
 * @param ops      Communication ops vtable (must have send/recv)
 * @param ctx      Backend context
 * @return 0 on success, -1 on failure
 */
int cml_ring_allreduce(float* data, size_t count, int world_size, int rank,
                       DistReduceOp op, DistCommOps* ops, void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* CML_DISTRIBUTED_RING_ALLREDUCE_H */
