#ifndef CML_DISTRIBUTED_RING_ATTENTION_H
#define CML_DISTRIBUTED_RING_ATTENTION_H

#include "tensor/tensor.h"
#include "nn/llm_ops.h" /* CMLGQAConfig */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Ring (sequence-parallel) FlashAttention: the sequence is sharded across the default process
 * group, each rank holding its local Q,K,V shard [batch, shard_len, heads*head_dim] (shard_len
 * equal across ranks). KV shards rotate around the ring while every rank folds each one into an
 * online softmax, so each rank's local Q attends the FULL global sequence without materializing
 * the global score matrix. config->causal masks by global position, making this the prefill
 * path. With world_size <= 1 it falls back to the local flash kernel. Returns the local output
 * [batch, shard_len, num_heads*head_dim] on CPU f32, or NULL on error.
 * @param Q (Tensor*)
 * @param K (Tensor*)
 * @param V (Tensor*)
 * @param config (const CMLGQAConfig*)
 * @return (Tensor*)
 */
Tensor* cml_ring_attention_forward(Tensor* Q, Tensor* K, Tensor* V, const CMLGQAConfig* config);

/**
 * Ring attention decode step: the new query shard @p Q (typically [batch, 1, num_heads*head_dim],
 * replicated on every rank) attends over the KV cache sharded as K,V across the ring. Every cached
 * token precedes the query, so no causal mask is applied regardless of config->causal. Returns the
 * query output [batch, seq_q, num_heads*head_dim] (identical on every rank), or NULL on error.
 * @param Q (Tensor*)
 * @param K (Tensor*)
 * @param V (Tensor*)
 * @param config (const CMLGQAConfig*)
 * @return (Tensor*)
 */
Tensor* cml_ring_attention_decode(Tensor* Q, Tensor* K, Tensor* V, const CMLGQAConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_DISTRIBUTED_RING_ATTENTION_H */
