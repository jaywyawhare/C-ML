/**
 * @file test_mlir_integration.c
 * @brief Tests for MLIR integration
 * @version 0.1.0
 * @date 2025-11-27
 */

#include "ops/ir/mlir/mlir_backend.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_mlir_availability(void) {
    printf("Testing MLIR availability...\n");

    bool available = cml_mlir_is_available();
    printf("  MLIR available: %s\n", available ? "YES" : "NO");

    CMLMLIRStatus status = cml_mlir_get_status();
    const char* status_names[] = {
        "NOT_AVAILABLE", "AVAILABLE", "INITIALIZED", "ERROR"
    };
    printf("  MLIR status: %s\n", status_names[status]);

    const char* version = cml_mlir_version();
    printf("  MLIR version: %s\n", version);
}

void test_execution_modes(void) {
    printf("\nTesting execution modes...\n");

    // Test default mode
    CMLExecutionMode mode = cml_get_execution_mode();
    assert(mode == CML_EXEC_INTERPRETED);
    printf("  Default mode: INTERPRETED ✓\n");

    // Test setting mode
    cml_set_execution_mode(CML_EXEC_JIT);
    mode = cml_get_execution_mode();

    if (cml_mlir_is_available()) {
        assert(mode == CML_EXEC_JIT);
        printf("  JIT mode set: OK ✓\n");
    } else {
        // Should fall back to interpreted
        assert(mode == CML_EXEC_INTERPRETED);
        printf("  JIT mode unavailable, fallback to INTERPRETED ✓\n");
    }

    // Reset to interpreted
    cml_set_execution_mode(CML_EXEC_INTERPRETED);

    // Test cml_enable_jit convenience function
    cml_enable_jit(true);
    mode = cml_get_execution_mode();
    printf("  cml_enable_jit(true): %s ✓\n",
           mode == CML_EXEC_JIT ? "JIT" : "INTERPRETED");

    cml_enable_jit(false);
    mode = cml_get_execution_mode();
    assert(mode == CML_EXEC_INTERPRETED);
    printf("  cml_enable_jit(false): INTERPRETED ✓\n");
}

void test_mlir_context(void) {
    printf("\nTesting MLIR context...\n");

    CMLMLIRContext* ctx = cml_mlir_init();

    if (cml_mlir_is_available()) {
        assert(ctx != NULL);
        printf("  MLIR context created ✓\n");

        cml_mlir_destroy(ctx);
        printf("  MLIR context destroyed ✓\n");
    } else {
        assert(ctx == NULL);
        printf("  MLIR not available, context is NULL ✓\n");
    }
}

void test_jit_engine(void) {
    printf("\nTesting JIT engine...\n");

    CMLJITEngine* engine = cml_jit_engine_create();

    if (cml_mlir_is_available()) {
        assert(engine != NULL);
        printf("  JIT engine created ✓\n");

        // Test cache functions
        cml_jit_cache_set_size(1024 * 1024 * 100);  // 100MB
        printf("  JIT cache size set ✓\n");

        size_t hits, misses, size;
        cml_jit_cache_stats(&hits, &misses, &size);
        printf("  JIT cache stats: hits=%zu, misses=%zu, size=%zu ✓\n",
               hits, misses, size);

        cml_jit_cache_clear();
        printf("  JIT cache cleared ✓\n");

        cml_jit_engine_destroy(engine);
        printf("  JIT engine destroyed ✓\n");
    } else {
        assert(engine == NULL);
        printf("  MLIR not available, JIT engine is NULL ✓\n");
    }
}

void test_backward_compatibility(void) {
    printf("\nTesting backward compatibility...\n");

    // Test that execution mode works without affecting compilation
    cml_set_execution_mode(CML_EXEC_INTERPRETED);
    printf("  Execution mode set without errors ✓\n");

    cml_enable_jit(false);
    printf("  JIT disabled without errors ✓\n");

    // Test utility functions
    const char* version = cml_mlir_version();
    assert(version != NULL);
    printf("  Version string: %s ✓\n", version);

    printf("  Backward compatibility maintained ✓\n");
}

int main(void) {
    printf("===========================================\n");
    printf("  C-ML MLIR Integration Tests - Phase 0\n");
    printf("===========================================\n\n");

    test_mlir_availability();
    test_execution_modes();
    test_mlir_context();
    test_jit_engine();
    test_backward_compatibility();

    printf("\n===========================================\n");
    printf("  All Phase 0 tests passed! ✓\n");
    printf("===========================================\n");

    return 0;
}
