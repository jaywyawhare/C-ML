/*
 * delegate.c — Backend delegation for subgraph execution
 */

#include "torch/delegate.h"
#include "ops/ir/internal.h"
#include "ops/ir/context.h"
#include "ops/ir/execution.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

static TorchDelegateRegistry g_registry = {0};

static int builtin_cpu_compile(TorchDelegate* self, CMLGraph_t subgraph, void** blob_out,
                               size_t* blob_size_out) {
    (void)self;
    (void)subgraph;
    if (blob_out)
        *blob_out = NULL;
    if (blob_size_out)
        *blob_size_out = 0;
    return 0;
}

static int builtin_cpu_execute(TorchDelegate* self, const void* blob, size_t blob_size,
                               Tensor** inputs, int num_inputs, Tensor** outputs, int num_outputs) {
    (void)self;
    (void)blob;
    (void)blob_size;
    CMLDispatchContext* ctx = cml_dispatch_get_global();
    if (!ctx)
        return -1;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    return cml_dispatch_execute_on(ctx, CML_BACKEND_CPU_FALLBACK, ir, inputs, num_inputs, outputs,
                                   num_outputs);
}

static int builtin_gpu_execute(TorchDelegate* self, const void* blob, size_t blob_size,
                               Tensor** inputs, int num_inputs, Tensor** outputs, int num_outputs) {
    (void)blob;
    (void)blob_size;
    CMLDispatchContext* ctx = cml_dispatch_get_global();
    if (!ctx || !self)
        return -1;
    CMLGraph_t ir = cml_ir_get_or_create_context();
    return cml_dispatch_execute_on(ctx, self->backend, ir, inputs, num_inputs, outputs,
                                   num_outputs);
}

static TorchDelegate g_cpu_delegate = {
    .name    = "cpu",
    .backend = CML_BACKEND_CPU_FALLBACK,
    .compile = builtin_cpu_compile,
    .execute = builtin_cpu_execute,
};

static TorchDelegate g_cuda_delegate = {
    .name    = "cuda",
    .backend = CML_BACKEND_CUDA,
    .compile = builtin_cpu_compile,
    .execute = builtin_gpu_execute,
};

static TorchDelegate g_vulkan_delegate = {
    .name    = "vulkan",
    .backend = CML_BACKEND_VULKAN,
    .compile = builtin_cpu_compile,
    .execute = builtin_gpu_execute,
};

static void delegate_registry_init(void) {
    if (g_registry.capacity > 0)
        return;
    g_registry.capacity = 8;
    g_registry.entries  = calloc((size_t)g_registry.capacity, sizeof(TorchDelegate));
    if (!g_registry.entries) {
        g_registry.capacity = 0;
        return;
    }
    g_registry.entries[g_registry.count++] = g_cpu_delegate;
    g_registry.entries[g_registry.count++] = g_cuda_delegate;
    g_registry.entries[g_registry.count++] = g_vulkan_delegate;
}

TorchDelegateRegistry* torch_delegate_registry(void) {
    delegate_registry_init();
    return &g_registry;
}

void torch_delegate_registry_free(void) {
    free(g_registry.entries);
    g_registry.entries  = NULL;
    g_registry.count    = 0;
    g_registry.capacity = 0;
}

int torch_delegate_register(TorchDelegate* delegate) {
    if (!delegate || !delegate->execute)
        return -1;
    delegate_registry_init();
    if (!g_registry.entries)
        return -1;

    if (g_registry.count >= g_registry.capacity) {
        g_registry.capacity *= 2;
        TorchDelegate* tmp =
            realloc(g_registry.entries, (size_t)g_registry.capacity * sizeof(TorchDelegate));
        if (!tmp)
            return -1;
        g_registry.entries = tmp;
    }

    g_registry.entries[g_registry.count++] = *delegate;
    return 0;
}

int torch_delegate_unregister(const char* name) {
    if (!name)
        return -1;
    for (int i = 0; i < g_registry.count; i++) {
        if (strcmp(g_registry.entries[i].name, name) == 0) {
            if (g_registry.entries[i].destroy)
                g_registry.entries[i].destroy(&g_registry.entries[i]);
            memmove(&g_registry.entries[i], &g_registry.entries[i + 1],
                    (size_t)(g_registry.count - i - 1) * sizeof(TorchDelegate));
            g_registry.count--;
            return 0;
        }
    }
    return -1;
}

TorchDelegate* torch_delegate_find(const char* name) {
    if (!name)
        return NULL;
    delegate_registry_init();
    for (int i = 0; i < g_registry.count; i++) {
        if (strcmp(g_registry.entries[i].name, name) == 0)
            return &g_registry.entries[i];
    }
    return NULL;
}

TorchDelegate* torch_delegate_find_by_backend(CMLBackendType backend) {
    delegate_registry_init();
    for (int i = 0; i < g_registry.count; i++) {
        if (g_registry.entries[i].backend == backend)
            return &g_registry.entries[i];
    }
    return NULL;
}

TorchDelegate* torch_delegate_cpu(void) { return &g_cpu_delegate; }
TorchDelegate* torch_delegate_cuda(void) { return &g_cuda_delegate; }
TorchDelegate* torch_delegate_vulkan(void) { return &g_vulkan_delegate; }

bool torch_delegate_supports_op(const TorchDelegate* d, UOpType op) {
    if (!d)
        return false;
    if (!d->supported_ops || d->num_ops == 0)
        return true;
    for (int i = 0; i < d->num_ops; i++) {
        if (d->supported_ops[i] == op)
            return true;
    }
    return false;
}

static bool graph_all_ops_supported(CMLGraph_t ir, const TorchDelegate* d) {
    if (!ir || !d)
        return false;
    for (struct IRNode* node = ir->head; node; node = node->next) {
        if (!torch_delegate_supports_op(d, node->type))
            return false;
    }
    return true;
}

CMLBackendType torch_delegate_select_backend(CMLGraph_t ir) {
    CMLDispatchContext* ctx = cml_dispatch_get_global();
    if (!ctx)
        return CML_BACKEND_CPU_FALLBACK;

    delegate_registry_init();
    for (int i = 0; i < g_registry.count; i++) {
        TorchDelegate* d = &g_registry.entries[i];
        if (d->backend == CML_BACKEND_CPU_FALLBACK)
            continue;
        if (!cml_dispatch_backend_available(ctx, d->backend))
            continue;
        if (graph_all_ops_supported(ir, d))
            return d->backend;
    }
    return CML_BACKEND_CPU_FALLBACK;
}

TorchDelegatePlan* torch_delegate_partition_graph(CMLGraph_t ir, CMLBackendType preferred) {
    if (!ir)
        return NULL;

    TorchDelegatePlan* plan = calloc(1, sizeof(TorchDelegatePlan));
    if (!plan)
        return NULL;

    TorchDelegate* d = torch_delegate_find_by_backend(preferred);
    if (!d || !graph_all_ops_supported(ir, d)) {
        plan->fallback_graph = ir;
        plan->num_partitions = 0;
        return plan;
    }

    plan->num_partitions = 1;
    plan->partitions     = calloc(1, sizeof(TorchDelegatePartition));
    if (!plan->partitions) {
        free(plan);
        return NULL;
    }
    plan->partitions[0].delegate_index = 0;
    plan->partitions[0].subgraph         = ir;

    for (int i = 0; i < g_registry.count; i++) {
        if (g_registry.entries[i].backend == preferred) {
            plan->partitions[0].delegate_index = i;
            break;
        }
    }
    return plan;
}

void torch_delegate_plan_free(TorchDelegatePlan* plan) {
    if (!plan)
        return;
    free(plan->partitions);
    free(plan);
}

int torch_delegate_execute_plan(TorchDelegatePlan* plan, Tensor** inputs, int num_inputs,
                                Tensor** outputs, int num_outputs) {
    if (!plan)
        return -1;

    if (plan->num_partitions == 0) {
        CMLDispatchContext* ctx = cml_dispatch_get_global();
        if (!ctx)
            return -1;
        return cml_dispatch_execute(ctx, plan->fallback_graph, inputs, num_inputs, outputs,
                                    num_outputs);
    }

    if (!g_registry.entries ||
        plan->partitions[0].delegate_index < 0 ||
        plan->partitions[0].delegate_index >= g_registry.count)
        return -1;
    TorchDelegate* d = &g_registry.entries[plan->partitions[0].delegate_index];
    return d->execute(d, plan->partitions[0].blob, plan->partitions[0].blob_size, inputs,
                      num_inputs, outputs, num_outputs);
}
