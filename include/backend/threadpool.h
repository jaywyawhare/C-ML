/**
 * @file threadpool.h
 * @brief Thread pool for multithreaded operations
 *
 * Provides a thread pool for parallelizing elementwise operations
 * and fused kernels.
 */

#ifndef CML_CORE_THREADPOOL_H
#define CML_CORE_THREADPOOL_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Thread pool handle
 */
typedef struct ThreadPool ThreadPool;

/**
 * @brief Task function signature
 */
typedef void (*TaskFunc)(void* data, size_t start, size_t end);

/**
 * @brief Task data
 */
typedef struct {
    TaskFunc func;
    void* data;
    size_t total_size;
} Task;

/**
 * @brief Create thread pool
 *
 * @param num_threads Number of threads (0 for auto-detect)
 * @return Thread pool handle, or NULL on failure
 */
ThreadPool* threadpool_create(size_t num_threads);

/**
 * @brief Destroy thread pool
 */
void threadpool_destroy(ThreadPool* pool);

/**
 * @brief Submit task to thread pool
 *
 * @param pool Thread pool
 * @param task Task to execute
 * @return 0 on success, -1 on failure
 */
int threadpool_submit(ThreadPool* pool, Task* task);

/**
 * @brief Wait for all tasks to complete
 */
void threadpool_wait(ThreadPool* pool);

/**
 * @brief Get number of threads
 */
size_t threadpool_get_num_threads(ThreadPool* pool);

/**
 * @brief Get global thread pool
 */
ThreadPool* threadpool_get_global(void);

/**
 * @brief Set global thread pool
 */
void threadpool_set_global(ThreadPool* pool);

/**
 * @brief Parallel for loop
 *
 * @param pool Thread pool (NULL for global)
 * @param func Task function
 * @param data Task data
 * @param n Number of iterations
 */
void threadpool_parallel_for(ThreadPool* pool, TaskFunc func, void* data, size_t n);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_THREADPOOL_H
