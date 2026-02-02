/**
 * @file execution.h
 * @brief IR execution engine
 */

#ifndef CML_OPS_IR_EXECUTION_H
#define CML_OPS_IR_EXECUTION_H

#include "ops/ir/ir.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execute compiled IR
 */
int cml_ir_execute(CMLIR_t ir);

/**
 * @brief Execute IR using CPU fallback interpreter
 *
 * This is a simple scalar interpreter used when MLIR is not available
 * or as a fallback when other backends fail.
 *
 * @param ir IR context
 * @return 0 on success, negative on failure
 */
int cpu_execute_ir(CMLIR_t ir);

/**
 * @brief Execute IR graph up to target node (lazy evaluation)
 *
 * Executes the IR graph up to and including the target node.
 * This is used for lazy evaluation when data is accessed.
 *
 * @param ir IR context
 * @param target_node Target node to execute up to
 * @return 0 on success, negative on failure
 */
int cml_ir_execute_up_to(CMLIR_t ir, struct IRNode* target_node);

/**
 * @brief Print execution statistics (for debugging)
 */
void cml_print_exec_stats(void);

/**
 * @brief Reset execution statistics
 */
void cml_reset_exec_stats(void);

/**
 * @brief Allocate memory from buffer cache
 *
 * Uses cached buffers when available, otherwise allocates new memory.
 * Buffers are allocated in power-of-2 size buckets for efficient reuse.
 *
 * @param size Size in bytes
 * @return Pointer to zero-initialized memory, or NULL on failure
 */
void* cml_buffer_cache_alloc(size_t size);

/**
 * @brief Return memory to buffer cache
 *
 * Returns buffer to cache for future reuse. If cache is full, frees memory.
 *
 * @param ptr Pointer to memory (can be NULL)
 * @param size Size in bytes (used to determine bucket)
 */
void cml_buffer_cache_free(void* ptr, size_t size);

/**
 * @brief Print buffer cache statistics
 */
void cml_print_buffer_cache_stats(void);

/**
 * @brief Clean up the buffer cache (call at program exit)
 */
void cml_cleanup_buffer_cache(void);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_EXECUTION_H
