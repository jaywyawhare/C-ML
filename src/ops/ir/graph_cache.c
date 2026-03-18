#include "ops/ir/graph_cache.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include "ops/simd_utils.h"
#include "ops/simd_math.h"
#include "backend/blas.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#define CACHE_NUM_BUCKETS 64
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL

static CMLGraphCache* g_graph_cache = NULL;

static uint64_t hash_combine(uint64_t h, uint64_t val) {
    h ^= val;
    h *= FNV_PRIME;
    return h;
}

uint64_t cml_graph_compute_signature(CMLGraph_t ir) {
    if (!ir)
        return 0;

    uint64_t hash = FNV_OFFSET;

    struct IRNode* node = ir->head;
    while (node) {
        hash = hash_combine(hash, (uint64_t)node->type);

        if (node->output) {
            hash = hash_combine(hash, (uint64_t)node->output->ndim);
            for (int i = 0; i < node->output->ndim; i++) {
                hash = hash_combine(hash, (uint64_t)node->output->shape[i]);
            }
        }

        hash = hash_combine(hash, (uint64_t)node->num_inputs);

        node = node->next;
    }

    return hash;
}

CMLGraphCache* cml_graph_cache_create(size_t max_entries) {
    CMLGraphCache* cache = calloc(1, sizeof(CMLGraphCache));
    if (!cache)
        return NULL;

    cache->num_buckets = CACHE_NUM_BUCKETS;
    cache->buckets     = calloc(cache->num_buckets, sizeof(CMLGraphCacheEntry*));
    if (!cache->buckets) {
        free(cache);
        return NULL;
    }

    cache->max_entries = max_entries;
    cache->count       = 0;
    cache->timestamp   = 0;
    cache->hits        = 0;
    cache->misses      = 0;

    return cache;
}

void cml_graph_cache_destroy(CMLGraphCache* cache) {
    if (!cache)
        return;

    for (size_t i = 0; i < cache->num_buckets; i++) {
        CMLGraphCacheEntry* entry = cache->buckets[i];
        while (entry) {
            CMLGraphCacheEntry* next = entry->next;
            cml_free_execution_plan(entry->plan);
            free(entry);
            entry = next;
        }
    }

    free(cache->buckets);
    free(cache);
}

CMLExecutionPlan* cml_graph_cache_lookup(CMLGraphCache* cache, uint64_t signature) {
    if (!cache)
        return NULL;

    size_t bucket             = signature % cache->num_buckets;
    CMLGraphCacheEntry* entry = cache->buckets[bucket];

    while (entry) {
        if (entry->signature == signature && entry->plan && entry->plan->valid) {
            cache->hits++;
            entry->last_used = cache->timestamp++;
            return entry->plan;
        }
        entry = entry->next;
    }

    cache->misses++;
    return NULL;
}

static void evict_lru_entry(CMLGraphCache* cache) {
    if (!cache || cache->count == 0)
        return;

    uint64_t oldest_time             = UINT64_MAX;
    size_t oldest_bucket             = 0;
    CMLGraphCacheEntry* oldest_entry = NULL;
    CMLGraphCacheEntry* oldest_prev  = NULL;

    for (size_t b = 0; b < cache->num_buckets; b++) {
        CMLGraphCacheEntry* prev  = NULL;
        CMLGraphCacheEntry* entry = cache->buckets[b];
        while (entry) {
            if (entry->last_used < oldest_time) {
                oldest_time   = entry->last_used;
                oldest_bucket = b;
                oldest_entry  = entry;
                oldest_prev   = prev;
            }
            prev  = entry;
            entry = entry->next;
        }
    }

    if (oldest_entry) {
        if (oldest_prev) {
            oldest_prev->next = oldest_entry->next;
        } else {
            cache->buckets[oldest_bucket] = oldest_entry->next;
        }
        cml_free_execution_plan(oldest_entry->plan);
        free(oldest_entry);
        cache->count--;
    }
}

int cml_graph_cache_insert(CMLGraphCache* cache, uint64_t signature, CMLExecutionPlan* plan) {
    if (!cache || !plan)
        return -1;

    if (cache->count >= cache->max_entries) {
        evict_lru_entry(cache);
    }

    size_t bucket = signature % cache->num_buckets;

    CMLGraphCacheEntry* entry = calloc(1, sizeof(CMLGraphCacheEntry));
    if (!entry)
        return -1;

    entry->signature       = signature;
    entry->plan            = plan;
    entry->last_used       = cache->timestamp++;
    entry->next            = cache->buckets[bucket];
    cache->buckets[bucket] = entry;
    cache->count++;

    return 0;
}

static size_t count_nodes(CMLGraph_t ir) {
    size_t count        = 0;
    struct IRNode* node = ir->head;
    while (node) {
        count++;
        node = node->next;
    }
    return count;
}

CMLExecutionPlan* cml_create_execution_plan(CMLGraph_t ir) {
    if (!ir)
        return NULL;

    CMLExecutionPlan* plan = calloc(1, sizeof(CMLExecutionPlan));
    if (!plan)
        return NULL;

    plan->num_nodes = count_nodes(ir);
    if (plan->num_nodes == 0) {
        free(plan);
        return NULL;
    }

    plan->nodes        = calloc(plan->num_nodes, sizeof(struct IRNode*));
    plan->buffers      = calloc(plan->num_nodes, sizeof(float*));
    plan->buffer_sizes = calloc(plan->num_nodes, sizeof(size_t));

    if (!plan->nodes || !plan->buffers || !plan->buffer_sizes) {
        cml_free_execution_plan(plan);
        return NULL;
    }

    struct IRNode* node = ir->head;
    size_t idx          = 0;
    while (node && idx < plan->num_nodes) {
        plan->nodes[idx] = node;

        if (node->output && node->output->numel > 0) {
            plan->buffer_sizes[idx] = node->output->numel;
            plan->buffers[idx]      = aligned_alloc(32, node->output->numel * sizeof(float));
            if (!plan->buffers[idx]) {
                cml_free_execution_plan(plan);
                return NULL;
            }
        }

        idx++;
        node = node->next;
    }

    plan->signature = cml_graph_compute_signature(ir);
    plan->valid     = true;

    return plan;
}

void cml_free_execution_plan(CMLExecutionPlan* plan) {
    if (!plan)
        return;

    if (plan->buffers) {
        for (size_t i = 0; i < plan->num_nodes; i++) {
            if (plan->buffers[i]) {
                free(plan->buffers[i]);
            }
        }
        free(plan->buffers);
    }

    free(plan->nodes);
    free(plan->buffer_sizes);
    free(plan);
}

static int execute_node_fast(struct IRNode* node, float* out_buf) {
    if (!node || !out_buf)
        return -1;

    float* in1   = NULL;
    float* in2   = NULL;
    size_t in1_n = 0, in2_n = 0;
    size_t out_n = node->output ? node->output->numel : 0;

    if (node->num_inputs >= 1 && node->inputs[0]) {
        in1   = (float*)node->inputs[0]->data;
        in1_n = node->inputs[0]->numel;
    }
    if (node->num_inputs >= 2 && node->inputs[1]) {
        in2   = (float*)node->inputs[1]->data;
        in2_n = node->inputs[1]->numel;
    }

    switch (node->type) {
    case UOP_ADD:
        if (!in1 || !in2)
            return -1;
        if (in1_n == in2_n && in1_n == out_n) {
            simd_add_f32(in1, in2, out_buf, out_n);
        } else {
            for (size_t i = 0; i < out_n; i++) {
                out_buf[i] = in1[in1_n == 1 ? 0 : i % in1_n] + in2[in2_n == 1 ? 0 : i % in2_n];
            }
        }
        break;

    case UOP_SUB:
        if (!in1 || !in2)
            return -1;
        if (in1_n == in2_n && in1_n == out_n) {
            simd_sub_f32(in1, in2, out_buf, out_n);
        } else {
            for (size_t i = 0; i < out_n; i++) {
                out_buf[i] = in1[in1_n == 1 ? 0 : i % in1_n] - in2[in2_n == 1 ? 0 : i % in2_n];
            }
        }
        break;

    case UOP_MUL:
        if (!in1 || !in2)
            return -1;
        if (in1_n == in2_n && in1_n == out_n) {
            simd_mul_f32(in1, in2, out_buf, out_n);
        } else {
            for (size_t i = 0; i < out_n; i++) {
                out_buf[i] = in1[in1_n == 1 ? 0 : i % in1_n] * in2[in2_n == 1 ? 0 : i % in2_n];
            }
        }
        break;

    case UOP_DIV:
        if (!in1 || !in2)
            return -1;
        if (in1_n == in2_n && in1_n == out_n) {
            simd_div_f32(in1, in2, out_buf, out_n);
        } else {
            for (size_t i = 0; i < out_n; i++) {
                float denom = in2[in2_n == 1 ? 0 : i % in2_n];
                out_buf[i]  = in1[in1_n == 1 ? 0 : i % in1_n] / (denom != 0.0f ? denom : 1e-8f);
            }
        }
        break;

    case UOP_NEG:
        if (!in1)
            return -1;
        simd_neg_f32(in1, out_buf, out_n);
        break;

    case UOP_EXP:
        if (!in1)
            return -1;
        simd_exp_f32(in1, out_buf, out_n);
        break;

    case UOP_LOG:
        if (!in1)
            return -1;
        simd_log_f32(in1, out_buf, out_n);
        break;

    case UOP_SQRT:
        if (!in1)
            return -1;
        simd_sqrt_f32(in1, out_buf, out_n);
        break;

    case UOP_ABS:
        if (!in1)
            return -1;
        simd_abs_f32(in1, out_buf, out_n);
        break;

    case UOP_SIGMOID:
        if (!in1)
            return -1;
        simd_sigmoid_f32(in1, out_buf, out_n);
        break;

    case UOP_TANH:
        if (!in1)
            return -1;
        simd_tanh_f32(in1, out_buf, out_n);
        break;

    case UOP_MATMUL: {
        if (!in1 || !in2)
            return -1;
        Tensor* a = node->inputs[0];
        Tensor* b = node->inputs[1];
        if (a->ndim < 2 || b->ndim < 2)
            return -1;

        int M = a->shape[a->ndim - 2];
        int K = a->shape[a->ndim - 1];
        int N = b->shape[b->ndim - 1];

        memset(out_buf, 0, out_n * sizeof(float));

        // Try BLAS
        CMLBlasContext* blas = cml_blas_get_context();
        if (blas && blas->initialized) {
            if (cml_blas_sgemm(blas, in1, in2, out_buf, M, N, K, 1.0f, 0.0f) == 0) {
                break;
            }
        }

        // Fallback: blocked matmul
        const int BLOCK = 32;
        for (int i0 = 0; i0 < M; i0 += BLOCK) {
            for (int j0 = 0; j0 < N; j0 += BLOCK) {
                for (int k0 = 0; k0 < K; k0 += BLOCK) {
                    int i_end = (i0 + BLOCK < M) ? i0 + BLOCK : M;
                    int j_end = (j0 + BLOCK < N) ? j0 + BLOCK : N;
                    int k_end = (k0 + BLOCK < K) ? k0 + BLOCK : K;
                    for (int i = i0; i < i_end; i++) {
                        for (int k = k0; k < k_end; k++) {
                            float a_ik = in1[i * K + k];
                            for (int j = j0; j < j_end; j++) {
                                out_buf[i * N + j] += a_ik * in2[k * N + j];
                            }
                        }
                    }
                }
            }
        }
        break;
    }

    case UOP_SUM: {
        if (!in1)
            return -1;
        ReduceParams* params = (ReduceParams*)node->params;
        if (!params || params->num_dims == 0) {
            // Sum all
            float sum = 0.0f;
            for (size_t i = 0; i < in1_n; i++) {
                sum += in1[i];
            }
            out_buf[0] = sum;
        } else {
            memset(out_buf, 0, out_n * sizeof(float));
            for (size_t i = 0; i < in1_n; i++) {
                out_buf[i % out_n] += in1[i];
            }
        }
        break;
    }

    case UOP_MAX: {
        if (!in1 || !in2)
            return -1;
        for (size_t i = 0; i < out_n; i++) {
            float v1   = in1[in1_n == 1 ? 0 : i % in1_n];
            float v2   = in2[in2_n == 1 ? 0 : i % in2_n];
            out_buf[i] = v1 > v2 ? v1 : v2;
        }
        break;
    }

    case UOP_RESHAPE:
    case UOP_PERMUTE:
    case UOP_EXPAND:
    case UOP_STRIDE:
    case UOP_SLICE:
        if (!in1)
            return -1;
        if (in1_n == out_n) {
            memcpy(out_buf, in1, out_n * sizeof(float));
        } else if (in1_n == 1) {
            for (size_t i = 0; i < out_n; i++) {
                out_buf[i] = in1[0];
            }
        } else {
            for (size_t i = 0; i < out_n; i++) {
                out_buf[i] = in1[i % in1_n];
            }
        }
        break;

    case UOP_CMPLT:
        if (!in1 || !in2)
            return -1;
        for (size_t i = 0; i < out_n; i++) {
            float v1   = in1[in1_n == 1 ? 0 : i % in1_n];
            float v2   = in2[in2_n == 1 ? 0 : i % in2_n];
            out_buf[i] = v1 < v2 ? 1.0f : 0.0f;
        }
        break;

    case UOP_RECIP:
        if (!in1)
            return -1;
        for (size_t i = 0; i < out_n; i++) {
            float v    = in1[i % in1_n];
            out_buf[i] = 1.0f / (v != 0.0f ? v : 1e-8f);
        }
        break;

    case UOP_MEAN: {
        if (!in1)
            return -1;
        ReduceParams* params = (ReduceParams*)node->params;
        if (!params || params->num_dims == 0) {
            // Mean all
            float sum = 0.0f;
            for (size_t i = 0; i < in1_n; i++) {
                sum += in1[i];
            }
            out_buf[0] = sum / (float)in1_n;
        } else {
            memset(out_buf, 0, out_n * sizeof(float));
            for (size_t i = 0; i < in1_n; i++) {
                out_buf[i % out_n] += in1[i];
            }
            float scale = (float)in1_n / (float)out_n;
            for (size_t i = 0; i < out_n; i++) {
                out_buf[i] /= scale;
            }
        }
        break;
    }

    case UOP_MAX_REDUCE: {
        if (!in1)
            return -1;
        float max_val = in1[0];
        for (size_t i = 1; i < in1_n; i++) {
            if (in1[i] > max_val)
                max_val = in1[i];
        }
        out_buf[0] = max_val;
        break;
    }

    case UOP_WHERE: {
        if (node->num_inputs < 3)
            return -1;
        float* cond = (float*)node->inputs[0]->data;
        float* a    = (float*)node->inputs[1]->data;
        float* b    = (float*)node->inputs[2]->data;
        if (!cond || !a || !b)
            return -1;
        size_t cond_n = node->inputs[0]->numel;
        size_t a_n    = node->inputs[1]->numel;
        size_t b_n    = node->inputs[2]->numel;
        for (size_t i = 0; i < out_n; i++) {
            float c    = cond[cond_n == 1 ? 0 : i % cond_n];
            out_buf[i] = (c != 0.0f) ? a[a_n == 1 ? 0 : i % a_n] : b[b_n == 1 ? 0 : i % b_n];
        }
        break;
    }

    default:
        LOG_WARNING("Unhandled op in fast path: %d", node->type);
        return -1;
    }

    return 0;
}

int cml_execute_plan(CMLExecutionPlan* plan, Tensor** inputs, size_t num_inputs) {
    if (!plan || !plan->valid)
        return -1;

    (void)inputs;
    (void)num_inputs;

    for (size_t i = 0; i < plan->num_nodes; i++) {
        struct IRNode* node = plan->nodes[i];
        float* out_buf      = plan->buffers[i];

        int ret = execute_node_fast(node, out_buf);
        if (ret != 0) {
            LOG_ERROR("Failed to execute node %zu (type %d)", i, node->type);
            return -1;
        }

        if (node->output) {
            node->output->data        = out_buf;
            node->output->is_executed = true;
        }
    }

    return 0;
}

CMLGraphCache* cml_get_graph_cache(void) {
    if (!g_graph_cache) {
        g_graph_cache = cml_graph_cache_create(32);
    }
    return g_graph_cache;
}

void cml_graph_cache_print_stats(CMLGraphCache* cache) {
    if (!cache) {
        cache = g_graph_cache;
    }
    if (!cache) {
        printf("Graph cache not initialized\n");
        return;
    }

    printf("Graph Cache Stats:\n");
    printf("  Entries: %zu / %zu\n", cache->count, cache->max_entries);
    printf("  Hits: %zu\n", cache->hits);
    printf("  Misses: %zu\n", cache->misses);
    if (cache->hits + cache->misses > 0) {
        printf("  Hit rate: %.1f%%\n", 100.0 * cache->hits / (cache->hits + cache->misses));
    }
}
