#ifndef CML_TENSOR_SHARD_H
#define CML_TENSOR_SHARD_H

#include "tensor/tensor.h"
#include "backend/device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLShardSpec {
    int axis;
    int num_shards;
    DeviceType* devices;
    int* shard_sizes;
} CMLShardSpec;

typedef struct CMLShardedTensor {
    Tensor** shards;
    int num_shards;
    int axis;
    int* shape;
    int ndim;
    DeviceType* devices;
} CMLShardedTensor;

CMLShardedTensor* tensor_shard(Tensor* t, DeviceType* devices, int num_devices, int axis);
CMLShardedTensor* tensor_shard_with_sizes(Tensor* t, DeviceType* devices, int num_devices, int axis, int* sizes);
Tensor* tensor_unshard(CMLShardedTensor* st);
void sharded_tensor_free(CMLShardedTensor* st);

CMLShardedTensor* sharded_matmul(CMLShardedTensor* a, CMLShardedTensor* b);
CMLShardedTensor* sharded_add(CMLShardedTensor* a, CMLShardedTensor* b);
CMLShardedTensor* sharded_allreduce_sum(CMLShardedTensor* st);

CMLShardedTensor* tensor_replicate(Tensor* t, DeviceType* devices, int num_devices);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_SHARD_H
