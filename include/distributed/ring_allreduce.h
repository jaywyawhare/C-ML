#ifndef CML_DISTRIBUTED_RING_ALLREDUCE_H
#define CML_DISTRIBUTED_RING_ALLREDUCE_H

#include "distributed/distributed.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cml_ring_allreduce(float* data, size_t count, int world_size, int rank, DistReduceOp op,
                       DistCommOps* ops, void* ctx);

/**
 * Full-duplex ring step shared by the ring collectives: send @p send_t to the right neighbour
 * while concurrently receiving @p recv_t from the left (disjoint buffers, deadlock-free), with
 * a serial fallback if no thread can be spawned. @p tag pairs each send with its matching recv.
 * @param ops (DistCommOps*)
 * @param send_t (Tensor*)
 * @param right (int)
 * @param recv_t (Tensor*)
 * @param left (int)
 * @param tag (int)
 * @param ctx (void*)
 * @return (int)
 */
int cml_ring_sendrecv(DistCommOps* ops, Tensor* send_t, int right, Tensor* recv_t, int left,
                      int tag, void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* CML_DISTRIBUTED_RING_ALLREDUCE_H */
