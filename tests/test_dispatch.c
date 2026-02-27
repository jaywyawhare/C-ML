/**
 * @file test_dispatch.c
 * @brief Unit tests for the unified dispatch layer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cml.h"
#include "ops/ir/dispatch.h"
#include "ops/ir/ir.h"
#include "ops/ir/context.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"

// Test counters
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        printf("FAIL\n"); \
    } \
} while(0)

// Test: Dispatch Context Creation

static int test_dispatch_create(void) {
    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) return 0;

    // Check initial state
    if (ctx->initialized) {
        cml_dispatch_free(ctx);
        return 0;
    }

    // CPU fallback should always be available
    if (ctx->backends[CML_BACKEND_CPU_FALLBACK].status != CML_BACKEND_STATUS_AVAILABLE) {
        cml_dispatch_free(ctx);
        return 0;
    }

    cml_dispatch_free(ctx);
    return 1;
}

// Test: Dispatch Initialization

static int test_dispatch_init(void) {
    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) return 0;

    int result = cml_dispatch_init(ctx);
    if (result != 0) {
        cml_dispatch_free(ctx);
        return 0;
    }

    // Should be initialized now
    if (!ctx->initialized) {
        cml_dispatch_free(ctx);
        return 0;
    }

    // Should have an active backend
    if (ctx->active >= CML_BACKEND_COUNT) {
        cml_dispatch_free(ctx);
        return 0;
    }

    cml_dispatch_free(ctx);
    return 1;
}

// Test: Backend Detection

static int test_backend_detection(void) {
    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) return 0;

    int num_backends = cml_dispatch_detect_backends(ctx);

    // At least CPU fallback should be detected
    if (num_backends < 1) {
        cml_dispatch_free(ctx);
        return 0;
    }

    // CPU fallback must be available
    if (!cml_dispatch_backend_available(ctx, CML_BACKEND_CPU_FALLBACK)) {
        cml_dispatch_free(ctx);
        return 0;
    }

    cml_dispatch_free(ctx);
    return 1;
}

// Test: Backend Names

static int test_backend_names(void) {
    const char* name;

    name = cml_dispatch_backend_name(CML_BACKEND_CPU_FALLBACK);
    if (!name || strlen(name) == 0) return 0;

    name = cml_dispatch_backend_name(CML_BACKEND_CPU_LLVM);
    if (!name || strlen(name) == 0) return 0;

    name = cml_dispatch_backend_name(CML_BACKEND_CUDA);
    if (!name || strlen(name) == 0) return 0;

    name = cml_dispatch_backend_name(CML_BACKEND_ROCM);
    if (!name || strlen(name) == 0) return 0;

    // Invalid backend should return "Unknown"
    name = cml_dispatch_backend_name(CML_BACKEND_COUNT + 1);
    if (!name) return 0;

    return 1;
}

// Test: Set Preferred Backend

static int test_set_preferred(void) {
    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) return 0;

    cml_dispatch_detect_backends(ctx);

    // Setting CPU fallback should always succeed
    int result = cml_dispatch_set_preferred(ctx, CML_BACKEND_CPU_FALLBACK);
    if (result != 0) {
        cml_dispatch_free(ctx);
        return 0;
    }

    if (ctx->preferred != CML_BACKEND_CPU_FALLBACK) {
        cml_dispatch_free(ctx);
        return 0;
    }

    cml_dispatch_free(ctx);
    return 1;
}

// Test: Global Context Singleton

static int test_global_context(void) {
    CMLDispatchContext* ctx1 = cml_dispatch_get_global();
    if (!ctx1) return 0;

    CMLDispatchContext* ctx2 = cml_dispatch_get_global();
    if (!ctx2) return 0;

    // Should be the same instance
    if (ctx1 != ctx2) return 0;

    // Should be initialized
    if (!ctx1->initialized) return 0;

    return 1;
}

// Test: Backend Info

static int test_backend_info(void) {
    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) return 0;

    cml_dispatch_detect_backends(ctx);

    const CMLBackendInfo* info = cml_dispatch_get_backend_info(ctx, CML_BACKEND_CPU_FALLBACK);
    if (!info) {
        cml_dispatch_free(ctx);
        return 0;
    }

    if (info->type != CML_BACKEND_CPU_FALLBACK) {
        cml_dispatch_free(ctx);
        return 0;
    }

    if (!info->name || strlen(info->name) == 0) {
        cml_dispatch_free(ctx);
        return 0;
    }

    // Invalid backend should return NULL
    info = cml_dispatch_get_backend_info(ctx, CML_BACKEND_COUNT + 1);
    if (info != NULL) {
        cml_dispatch_free(ctx);
        return 0;
    }

    cml_dispatch_free(ctx);
    return 1;
}

// Test: Best Backend Selection

static int test_best_backend(void) {
    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) return 0;

    cml_dispatch_detect_backends(ctx);

    CMLBackendType best = cml_dispatch_get_best_backend(ctx);

    // Must be a valid backend
    if (best >= CML_BACKEND_COUNT) {
        cml_dispatch_free(ctx);
        return 0;
    }

    // Must be available
    if (!cml_dispatch_backend_available(ctx, best)) {
        cml_dispatch_free(ctx);
        return 0;
    }

    cml_dispatch_free(ctx);
    return 1;
}

// Test: Simple IR Execution via Dispatch

static int test_dispatch_execute_simple(void) {
    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) return 0;

    cml_dispatch_init(ctx);

    // Create a simple IR with add operation
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) {
        cml_dispatch_free(ctx);
        return 0;
    }

    // Set as global context for tensor operations
    cml_ir_set_global_context(ir);

    // Create input tensors
    Tensor* a = tensor_empty_2d(2, 2);
    Tensor* b = tensor_empty_2d(2, 2);

    if (!a || !b) {
        if (a) tensor_free(a);
        if (b) tensor_free(b);
        cml_ir_free(ir);
        cml_dispatch_free(ctx);
        return 0;
    }

    // Fill with test data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int i = 0; i < 4; i++) {
        a_data[i] = (float)i;
        b_data[i] = (float)(i + 1);
    }

    // Add operation (adds to global IR context)
    Tensor* result = tensor_add(a, b);
    if (!result) {
        tensor_free(a);
        tensor_free(b);
        cml_ir_free(ir);
        cml_dispatch_free(ctx);
        return 0;
    }

    // Execute via dispatch
    int exec_result = cml_dispatch_execute(ctx, ir, NULL, 0, NULL, 0);

    // Check execution succeeded
    int success = (exec_result == 0);

    // Verify result if execution succeeded
    if (success && result->data) {
        float* r_data = (float*)result->data;
        // Expected: [0+1, 1+2, 2+3, 3+4] = [1, 3, 5, 7]
        if (r_data[0] != 1.0f || r_data[1] != 3.0f ||
            r_data[2] != 5.0f || r_data[3] != 7.0f) {
            success = 0;
        }
    }

    tensor_free(a);
    tensor_free(b);
    cml_ir_free(ir);
    cml_dispatch_free(ctx);

    return success;
}

// Test: Environment Variable Backend Selection

static int test_env_backend_selection(void) {
    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) return 0;

    cml_dispatch_detect_backends(ctx);

    // Save original env
    char* original = getenv("CML_BACKEND");
    char* saved = NULL;
    if (original) {
        saved = strdup(original);
    }

    // Test setting via env
    setenv("CML_BACKEND", "fallback", 1);
    int result = cml_dispatch_set_from_env(ctx);

    // Restore original env
    if (saved) {
        setenv("CML_BACKEND", saved, 1);
        free(saved);
    } else {
        unsetenv("CML_BACKEND");
    }

    if (result != 0) {
        cml_dispatch_free(ctx);
        return 0;
    }

    if (ctx->preferred != CML_BACKEND_CPU_FALLBACK) {
        cml_dispatch_free(ctx);
        return 0;
    }

    cml_dispatch_free(ctx);
    return 1;
}

// Test: Statistics Tracking

static int test_statistics(void) {
    CMLDispatchContext* ctx = cml_dispatch_create();
    if (!ctx) return 0;

    cml_dispatch_init(ctx);

    // Initial stats should be zero
    if (ctx->executions_total != 0 ||
        ctx->cache_hits != 0 ||
        ctx->cache_misses != 0) {
        cml_dispatch_free(ctx);
        return 0;
    }

    // Create and execute a simple IR
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    cml_ir_set_global_context(ir);

    Tensor* a = tensor_empty_2d(2, 2);
    Tensor* b = tensor_empty_2d(2, 2);

    if (ir && a && b) {
        tensor_add(a, b);
        cml_dispatch_execute(ctx, ir, NULL, 0, NULL, 0);
    }

    // Execution count should increase
    int success = (ctx->executions_total >= 1);

    if (a) tensor_free(a);
    if (b) tensor_free(b);
    if (ir) cml_ir_free(ir);
    cml_dispatch_free(ctx);

    return success;
}

// Main

int main(void) {
    printf("\n=== Dispatch Layer Unit Tests ===\n\n");

    TEST(dispatch_create);
    TEST(dispatch_init);
    TEST(backend_detection);
    TEST(backend_names);
    TEST(set_preferred);
    TEST(global_context);
    TEST(backend_info);
    TEST(best_backend);
    TEST(dispatch_execute_simple);
    TEST(env_backend_selection);
    TEST(statistics);

    printf("\n=================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("=================================\n\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
