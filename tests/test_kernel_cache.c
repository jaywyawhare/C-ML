#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cml.h"
#include "ops/ir/kernel_cache.h"
#include "ops/ir/ir.h"
#include "ops/ir/context.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include "core/logging.h"

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


static int test_cache_create(void) {
    CMLKernelCache* cache = cml_kernel_cache_create(100);
    if (!cache) return 0;

    cml_kernel_cache_free(cache);
    return 1;
}


static int test_cache_create_zero(void) {
    // Zero means unlimited
    CMLKernelCache* cache = cml_kernel_cache_create(0);
    if (!cache) return 0;

    cml_kernel_cache_free(cache);
    return 1;
}


static int test_hash_computation(void) {
    CMLGraph_t ir = cml_ir_new(IR_TARGET_C);
    if (!ir) return 0;

    cml_ir_set_global_context(ir);

    Tensor* a = tensor_empty_2d(2, 2);
    Tensor* b = tensor_empty_2d(2, 2);

    if (!a || !b) {
        if (a) tensor_free(a);
        if (b) tensor_free(b);
        cml_ir_free(ir);
        return 0;
    }

    tensor_add(a, b);

    Tensor* inputs[] = {a, b};
    uint64_t hash1 = cml_kernel_cache_compute_hash(ir, inputs, 2, CML_KERNEL_CPU_LLVM);
    uint64_t hash2 = cml_kernel_cache_compute_hash(ir, inputs, 2, CML_KERNEL_CPU_LLVM);

    // Same inputs should produce same hash
    int success = (hash1 == hash2);

    // Different backend should produce different hash
    uint64_t hash3 = cml_kernel_cache_compute_hash(ir, inputs, 2, CML_KERNEL_CUDA);
    if (hash1 == hash3) success = 0;

    tensor_free(a);
    tensor_free(b);
    cml_ir_free(ir);

    return success;
}


static int test_insert_lookup(void) {
    CMLKernelCache* cache = cml_kernel_cache_create(100);
    if (!cache) return 0;

    uint64_t hash = 0x123456789ABCDEF0ULL;
    void* dummy_kernel = (void*)0xDEADBEEF;

    // Insert
    int result = cml_kernel_cache_insert(cache, hash, CML_KERNEL_CPU_LLVM, dummy_kernel, 1024);
    if (result != 0) {
        cml_kernel_cache_free(cache);
        return 0;
    }

    // Lookup
    CMLKernelEntry* entry = cml_kernel_cache_lookup(cache, hash);
    if (!entry) {
        cml_kernel_cache_free(cache);
        return 0;
    }

    // Verify entry
    if (entry->hash != hash ||
        entry->backend != CML_KERNEL_CPU_LLVM ||
        entry->compiled != dummy_kernel) {
        cml_kernel_cache_free(cache);
        return 0;
    }

    cml_kernel_cache_free(cache);
    return 1;
}


static int test_cache_miss(void) {
    CMLKernelCache* cache = cml_kernel_cache_create(100);
    if (!cache) return 0;

    // Lookup without insert should return NULL
    CMLKernelEntry* entry = cml_kernel_cache_lookup(cache, 0x12345678);
    if (entry != NULL) {
        cml_kernel_cache_free(cache);
        return 0;
    }

    cml_kernel_cache_free(cache);
    return 1;
}


static int test_cache_clear(void) {
    CMLKernelCache* cache = cml_kernel_cache_create(100);
    if (!cache) return 0;

    // Insert some entries
    cml_kernel_cache_insert(cache, 0x1111, CML_KERNEL_CPU_LLVM, (void*)0x1, 100);
    cml_kernel_cache_insert(cache, 0x2222, CML_KERNEL_CPU_LLVM, (void*)0x2, 100);
    cml_kernel_cache_insert(cache, 0x3333, CML_KERNEL_CPU_LLVM, (void*)0x3, 100);

    // Verify entries exist
    if (!cml_kernel_cache_lookup(cache, 0x1111) ||
        !cml_kernel_cache_lookup(cache, 0x2222) ||
        !cml_kernel_cache_lookup(cache, 0x3333)) {
        cml_kernel_cache_free(cache);
        return 0;
    }

    // Clear (use internal function that takes cache pointer)
    kernel_cache_clear(cache);

    // Verify entries are gone
    if (cml_kernel_cache_lookup(cache, 0x1111) ||
        cml_kernel_cache_lookup(cache, 0x2222) ||
        cml_kernel_cache_lookup(cache, 0x3333)) {
        cml_kernel_cache_free(cache);
        return 0;
    }

    cml_kernel_cache_free(cache);
    return 1;
}


static int test_lru_eviction(void) {
    // Create cache with max 3 entries
    CMLKernelCache* cache = cml_kernel_cache_create(3);
    if (!cache) return 0;

    // Insert 3 entries
    cml_kernel_cache_insert(cache, 0x1111, CML_KERNEL_CPU_LLVM, (void*)0x1, 100);
    cml_kernel_cache_insert(cache, 0x2222, CML_KERNEL_CPU_LLVM, (void*)0x2, 100);
    cml_kernel_cache_insert(cache, 0x3333, CML_KERNEL_CPU_LLVM, (void*)0x3, 100);

    // Access first entry to make it recently used
    cml_kernel_cache_lookup(cache, 0x1111);

    // Insert 4th entry - should evict 0x2222 (least recently used)
    cml_kernel_cache_insert(cache, 0x4444, CML_KERNEL_CPU_LLVM, (void*)0x4, 100);

    // 0x1111 should still exist (was accessed)
    int has_1111 = (cml_kernel_cache_lookup(cache, 0x1111) != NULL);
    // 0x3333 should still exist
    int has_3333 = (cml_kernel_cache_lookup(cache, 0x3333) != NULL);
    // 0x4444 should exist (just inserted)
    int has_4444 = (cml_kernel_cache_lookup(cache, 0x4444) != NULL);
    // 0x2222 should be evicted
    int has_2222 = (cml_kernel_cache_lookup(cache, 0x2222) != NULL);

    cml_kernel_cache_free(cache);

    // 0x2222 should have been evicted
    return has_1111 && has_3333 && has_4444 && !has_2222;
}


static int test_cache_statistics(void) {
    CMLKernelCache* cache = cml_kernel_cache_create(100);
    if (!cache) return 0;

    // Insert entry
    cml_kernel_cache_insert(cache, 0x1111, CML_KERNEL_CPU_LLVM, (void*)0x1, 1024);

    // Hit
    cml_kernel_cache_lookup(cache, 0x1111);
    // Miss
    cml_kernel_cache_lookup(cache, 0x9999);

    size_t hits, misses, count, memory;
    // Use internal function that takes cache pointer
    kernel_cache_stats(cache, &hits, &misses, &count, &memory);

    int success = (hits == 1 && misses == 1 && count == 1 && memory == 1024);

    cml_kernel_cache_free(cache);
    return success;
}


static int test_multiple_backends(void) {
    CMLKernelCache* cache = cml_kernel_cache_create(100);
    if (!cache) return 0;

    // Same IR hash but different backends should have different cache entries
    // The hash includes the backend type, so they'll be different hashes
    uint64_t hash_llvm = 0x1111;
    uint64_t hash_cuda = 0x2222;

    cml_kernel_cache_insert(cache, hash_llvm, CML_KERNEL_CPU_LLVM, (void*)0x1, 100);
    cml_kernel_cache_insert(cache, hash_cuda, CML_KERNEL_CUDA, (void*)0x2, 100);

    CMLKernelEntry* entry_llvm = cml_kernel_cache_lookup(cache, hash_llvm);
    CMLKernelEntry* entry_cuda = cml_kernel_cache_lookup(cache, hash_cuda);

    int success = 1;
    if (!entry_llvm || entry_llvm->backend != CML_KERNEL_CPU_LLVM) success = 0;
    if (!entry_cuda || entry_cuda->backend != CML_KERNEL_CUDA) success = 0;

    cml_kernel_cache_free(cache);
    return success;
}


static int test_entry_update(void) {
    CMLKernelCache* cache = cml_kernel_cache_create(100);
    if (!cache) return 0;

    // Insert initial entry
    cml_kernel_cache_insert(cache, 0x1111, CML_KERNEL_CPU_LLVM, (void*)0x1, 100);

    // Update with same hash
    cml_kernel_cache_insert(cache, 0x1111, CML_KERNEL_CPU_LLVM, (void*)0x2, 200);

    // Lookup should return updated entry
    CMLKernelEntry* entry = cml_kernel_cache_lookup(cache, 0x1111);

    int success = (entry && entry->compiled == (void*)0x2 && entry->memory_size == 200);

    cml_kernel_cache_free(cache);
    return success;
}


static int test_large_cache(void) {
    CMLKernelCache* cache = cml_kernel_cache_create(1000);
    if (!cache) return 0;

    // Insert many entries
    for (int i = 0; i < 500; i++) {
        uint64_t hash = (uint64_t)i * 0x1234567890ABCDEFULL;
        cml_kernel_cache_insert(cache, hash, CML_KERNEL_CPU_LLVM, (void*)(uintptr_t)(i + 1), 64);
    }

    // Verify some entries
    int success = 1;
    for (int i = 0; i < 500; i += 50) {
        uint64_t hash = (uint64_t)i * 0x1234567890ABCDEFULL;
        CMLKernelEntry* entry = cml_kernel_cache_lookup(cache, hash);
        if (!entry || entry->compiled != (void*)(uintptr_t)(i + 1)) {
            success = 0;
            break;
        }
    }

    cml_kernel_cache_free(cache);
    return success;
}


int main(void) {
    printf("\nKernel Cache Unit Tests\n\n");

    TEST(cache_create);
    TEST(cache_create_zero);
    TEST(hash_computation);
    TEST(insert_lookup);
    TEST(cache_miss);
    TEST(cache_clear);
    TEST(lru_eviction);
    TEST(cache_statistics);
    TEST(multiple_backends);
    TEST(entry_update);
    TEST(large_cache);

    printf("\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
