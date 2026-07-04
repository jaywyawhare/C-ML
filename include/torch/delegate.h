/*
 * delegate.h — Backend delegation for subgraph execution
 *
 * ExecuTorch backend-delegate analogue: hardware vendors register AOT compilers
 * and runtime executors; the partitioner routes compatible subgraphs to delegates
 * with CPU fallback for the rest.
 */

#ifndef CML_TORCH_DELEGATE_H
#define CML_TORCH_DELEGATE_H

#include "core/export.h"
#include "ops/ir/dispatch.h"
#include "ops/ir/ir.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

typedef struct TorchDelegate {
    char             name[64];
    CMLBackendType   backend;
    UOpType*         supported_ops;
    int              num_ops;
    void*            user_data;

    /* AOT: lower a subgraph to a backend-specific blob */
    int (*compile)(struct TorchDelegate* self, CMLGraph_t subgraph,
                   void** blob_out, size_t* blob_size_out);

    /* Runtime: execute a previously compiled blob (required for registration). */
    int (*execute)(struct TorchDelegate* self, const void* blob, size_t blob_size,
                   Tensor** inputs, int num_inputs,
                   Tensor** outputs, int num_outputs);

    /* Optional teardown for user_data / backend resources. */
    void (*destroy)(struct TorchDelegate* self);
} TorchDelegate;

typedef struct TorchDelegateRegistry {
    TorchDelegate* entries;
    int            count;
    int            capacity;
} TorchDelegateRegistry;

typedef struct TorchDelegatePartition {
    int            delegate_index;
    CMLGraph_t     subgraph;
    void*          blob;
    size_t         blob_size;
    int*           node_indices;
    int            num_nodes;
} TorchDelegatePartition;

typedef struct TorchDelegatePlan {
    TorchDelegatePartition* partitions;
    int                     num_partitions;
    CMLGraph_t              fallback_graph;
} TorchDelegatePlan;

CML_API TorchDelegateRegistry* torch_delegate_registry(void);
CML_API void                   torch_delegate_registry_free(void);

CML_API int  torch_delegate_register(TorchDelegate* delegate);
CML_API int  torch_delegate_unregister(const char* name);
CML_API TorchDelegate* torch_delegate_find(const char* name);
CML_API TorchDelegate* torch_delegate_find_by_backend(CMLBackendType backend);

/* Built-in delegates wrapping C-ML dispatch backends */
CML_API TorchDelegate* torch_delegate_cpu(void);
CML_API TorchDelegate* torch_delegate_cuda(void);
CML_API TorchDelegate* torch_delegate_vulkan(void);

/* Returns false when d is NULL. When supported_ops is NULL/empty, all ops match. */
CML_API bool torch_delegate_supports_op(const TorchDelegate* d, UOpType op);

CML_API TorchDelegatePlan* torch_delegate_partition_graph(CMLGraph_t ir,
                                                          CMLBackendType preferred);
CML_API void               torch_delegate_plan_free(TorchDelegatePlan* plan);

CML_API int torch_delegate_execute_plan(TorchDelegatePlan* plan,
                                        Tensor** inputs, int num_inputs,
                                        Tensor** outputs, int num_outputs);

CML_API CMLBackendType torch_delegate_select_backend(CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif /* CML_TORCH_DELEGATE_H */
