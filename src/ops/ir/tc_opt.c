#include "ops/ir/tc_opt.h"
#include "ops/ir/pattern_matcher.h"
#include "ops/ir/internal.h"
#include "ops/ir/gpu/wmma.h"
#include "ops/ir/gpu/amx.h"
#include "ops/ir/gpu/xmx.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>

static CMLTCConfig g_tc_config = {
    .min_m = CML_TC_DEFAULT_MIN_DIM,
    .min_n = CML_TC_DEFAULT_MIN_DIM,
    .min_k = CML_TC_DEFAULT_MIN_DIM,
    .allow_padding = true,
    .prefer_fp16 = false,
};

static atomic_int g_tc_counter = 0;

static char* tc_unique_name(void) {
    int id = atomic_fetch_add(&g_tc_counter, 1);
    char* name = malloc(32);
    if (name)
        snprintf(name, 32, "_tc%d", id);
    return name;
}

void cml_tc_set_config(CMLTCConfig* config) {
    if (!config) return;
    g_tc_config = *config;
}

CMLTCConfig cml_tc_get_config(void) {
    return g_tc_config;
}

bool cml_tc_available(void) {
    return cml_wmma_available() || cml_amx_available() || cml_xmx_available();
}

typedef enum {
    TC_HW_NONE,
    TC_HW_WMMA,
    TC_HW_AMX,
    TC_HW_XMX,
} CMLTCHardware;

static CMLTCHardware tc_detect_hardware(void) {
    if (cml_wmma_available()) return TC_HW_WMMA;
    if (cml_amx_available())  return TC_HW_AMX;
    if (cml_xmx_available())  return TC_HW_XMX;
    return TC_HW_NONE;
}

static void tc_get_tile_size(CMLTCHardware hw, int* tile_m, int* tile_n, int* tile_k) {
    switch (hw) {
    case TC_HW_WMMA:
        *tile_m = 16; *tile_n = 16; *tile_k = 16;
        break;
    case TC_HW_AMX:
        *tile_m = 32; *tile_n = 32; *tile_k = 32;
        break;
    case TC_HW_XMX:
        *tile_m = 8; *tile_n = 16; *tile_k = 8;
        break;
    default:
        *tile_m = 16; *tile_n = 16; *tile_k = 16;
        break;
    }
}

static int round_up(int val, int multiple) {
    return ((val + multiple - 1) / multiple) * multiple;
}

static bool dims_tc_compatible(int m, int n, int k, const CMLTCConfig* cfg) {
    if (m < cfg->min_m || n < cfg->min_n || k < cfg->min_k)
        return false;
    if (m % 16 == 0 && n % 16 == 0 && k % 16 == 0)
        return true;
    return cfg->allow_padding;
}

static void extract_matmul_dims(struct IRNode* node, int* m, int* n, int* k) {
    *m = 0; *n = 0; *k = 0;
    if (!node || node->num_inputs < 2) return;
    if (!node->input_shapes || !node->input_ndims) return;

    int ndim_a = node->input_ndims[0];
    int ndim_b = node->input_ndims[1];
    if (ndim_a < 2 || ndim_b < 2) return;

    int* shape_a = node->input_shapes[0];
    int* shape_b = node->input_shapes[1];
    if (!shape_a || !shape_b) return;

    *m = shape_a[ndim_a - 2];
    *k = shape_a[ndim_a - 1];
    *n = shape_b[ndim_b - 1];
}

static struct IRNode* create_pad_node(struct IRNode* src_node, const char* src_name,
                                      int* padded_shape, int ndim) {
    struct IRNode* pad = calloc(1, sizeof(struct IRNode));
    if (!pad) return NULL;

    pad->type = UOP_PAD;
    pad->num_inputs = 1;
    pad->input_names = malloc(sizeof(char*));
    if (!pad->input_names) { free(pad); return NULL; }
    pad->input_names[0] = strdup(src_name);
    pad->output_name = tc_unique_name();
    pad->output_ndim = ndim;
    pad->output_shape = malloc((size_t)ndim * sizeof(int));
    if (pad->output_shape)
        memcpy(pad->output_shape, padded_shape, (size_t)ndim * sizeof(int));

    (void)src_node;
    return pad;
}

static struct IRNode* create_wmma_node(const char* a_name, const char* b_name,
                                       int m, int n, int k,
                                       int* output_shape, int output_ndim) {
    struct IRNode* wmma = calloc(1, sizeof(struct IRNode));
    if (!wmma) return NULL;

    wmma->type = UOP_MATMUL;
    wmma->is_fused = true;
    wmma->fusion_type = FUSION_FMA;
    wmma->num_inputs = 2;
    wmma->input_names = malloc(2 * sizeof(char*));
    if (!wmma->input_names) { free(wmma); return NULL; }
    wmma->input_names[0] = strdup(a_name);
    wmma->input_names[1] = strdup(b_name);
    wmma->output_name = tc_unique_name();
    wmma->output_ndim = output_ndim;
    wmma->output_shape = malloc((size_t)output_ndim * sizeof(int));
    if (wmma->output_shape)
        memcpy(wmma->output_shape, output_shape, (size_t)output_ndim * sizeof(int));

    WMMAConfig* cfg = calloc(1, sizeof(WMMAConfig));
    if (cfg) {
        cml_wmma_select_config(m, n, k, cfg);
        wmma->params = cfg;
    }

    return wmma;
}

static struct IRNode* find_node_by_output(CMLGraph_t ir, const char* name) {
    if (!ir || !name) return NULL;
    struct IRNode* n = ir->head;
    while (n) {
        if (n->output_name && strcmp(n->output_name, name) == 0)
            return n;
        n = n->next;
    }
    return NULL;
}

static void replace_refs(CMLGraph_t ir, const char* old_name, const char* new_name) {
    if (!ir || !old_name || !new_name) return;
    struct IRNode* n = ir->head;
    while (n) {
        for (int i = 0; i < n->num_inputs; i++) {
            if (n->input_names[i] && strcmp(n->input_names[i], old_name) == 0) {
                free(n->input_names[i]);
                n->input_names[i] = strdup(new_name);
            }
        }
        n = n->next;
    }
}

static void insert_before(CMLGraph_t ir, struct IRNode* new_node, struct IRNode* before) {
    if (!ir || !new_node) return;
    new_node->next = NULL;

    if (!before || !ir->head) {
        if (ir->tail)
            ir->tail->next = new_node;
        else
            ir->head = new_node;
        ir->tail = new_node;
        ir->node_count++;
        return;
    }

    if (ir->head == before) {
        new_node->next = before;
        ir->head = new_node;
        ir->node_count++;
        return;
    }

    struct IRNode* prev = ir->head;
    while (prev && prev->next != before)
        prev = prev->next;

    if (prev) {
        new_node->next = before;
        prev->next = new_node;
    } else {
        ir->tail->next = new_node;
        ir->tail = new_node;
    }
    ir->node_count++;
}

static void unlink_node(CMLGraph_t ir, struct IRNode* node) {
    if (!ir || !node) return;

    if (ir->head == node) {
        ir->head = node->next;
        if (ir->tail == node) ir->tail = NULL;
        ir->node_count--;
        return;
    }

    struct IRNode* prev = ir->head;
    while (prev && prev->next != node)
        prev = prev->next;
    if (prev) {
        prev->next = node->next;
        if (ir->tail == node) ir->tail = prev;
        ir->node_count--;
    }
}

static void free_node(struct IRNode* node) {
    if (!node) return;
    if (node->input_names) {
        for (int i = 0; i < node->num_inputs; i++)
            free(node->input_names[i]);
        free(node->input_names);
    }
    free(node->output_name);
    free(node->output_shape);
    free(node->params);
    free(node->users);
    if (node->output) {
        node->output->ir_node = NULL;
        node->output->ir_context = NULL;
    }
    free(node);
}

static int rewrite_matmul_to_wmma(CMLGraph_t ir, struct IRNode* node) {
    int m, n, k;
    extract_matmul_dims(node, &m, &n, &k);
    if (m == 0 || n == 0 || k == 0)
        return 0;

    if (!dims_tc_compatible(m, n, k, &g_tc_config))
        return 0;

    int pm = round_up(m, 16);
    int pn = round_up(n, 16);
    int pk = round_up(k, 16);
    bool needs_pad = (pm != m || pn != n || pk != k);

    if (needs_pad && !g_tc_config.allow_padding)
        return 0;

    const char* a_name = node->input_names[0];
    const char* b_name = node->input_names[1];
    const char* final_a = a_name;
    const char* final_b = b_name;

    if (needs_pad && node->input_shapes && node->input_ndims) {
        int ndim_a = node->input_ndims[0];
        int ndim_b = node->input_ndims[1];

        if (pm != m || pk != k) {
            int* padded_a = malloc((size_t)ndim_a * sizeof(int));
            if (padded_a) {
                memcpy(padded_a, node->input_shapes[0], (size_t)ndim_a * sizeof(int));
                padded_a[ndim_a - 2] = pm;
                padded_a[ndim_a - 1] = pk;
                struct IRNode* pad_a = create_pad_node(NULL, a_name, padded_a, ndim_a);
                free(padded_a);
                if (pad_a) {
                    insert_before(ir, pad_a, node);
                    final_a = pad_a->output_name;
                }
            }
        }

        if (pk != k || pn != n) {
            int* padded_b = malloc((size_t)ndim_b * sizeof(int));
            if (padded_b) {
                memcpy(padded_b, node->input_shapes[1], (size_t)ndim_b * sizeof(int));
                padded_b[ndim_b - 2] = pk;
                padded_b[ndim_b - 1] = pn;
                struct IRNode* pad_b = create_pad_node(NULL, b_name, padded_b, ndim_b);
                free(padded_b);
                if (pad_b) {
                    insert_before(ir, pad_b, node);
                    final_b = pad_b->output_name;
                }
            }
        }
    }

    int out_ndim = node->output_ndim > 0 ? node->output_ndim : 2;
    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape) return 0;

    if (node->output_shape && node->output_ndim > 0) {
        memcpy(out_shape, node->output_shape, (size_t)out_ndim * sizeof(int));
        out_shape[out_ndim - 2] = pm;
        out_shape[out_ndim - 1] = pn;
    } else {
        out_shape[0] = pm;
        out_shape[1] = pn;
    }

    struct IRNode* wmma = create_wmma_node(final_a, final_b, pm, pn, pk,
                                           out_shape, out_ndim);
    free(out_shape);
    if (!wmma) return 0;

    insert_before(ir, wmma, node);

    if (needs_pad && node->output_shape) {
        int* slice_shape = malloc((size_t)out_ndim * sizeof(int));
        if (slice_shape) {
            memcpy(slice_shape, node->output_shape, (size_t)out_ndim * sizeof(int));
            struct IRNode* slice = calloc(1, sizeof(struct IRNode));
            if (slice) {
                slice->type = UOP_SHRINK;
                slice->num_inputs = 1;
                slice->input_names = malloc(sizeof(char*));
                if (slice->input_names) {
                    slice->input_names[0] = strdup(wmma->output_name);
                    slice->output_name = tc_unique_name();
                    slice->output_ndim = out_ndim;
                    slice->output_shape = slice_shape;
                    insert_before(ir, slice, node);

                    if (node->output_name)
                        replace_refs(ir, node->output_name, slice->output_name);
                    if (node->output && slice->output_name) {
                        node->output->ir_node = slice;
                        node->output->ir_context = ir;
                    }
                } else {
                    free(slice_shape);
                    free(slice);
                }
            } else {
                free(slice_shape);
            }
        }
    } else {
        if (node->output_name)
            replace_refs(ir, node->output_name, wmma->output_name);
        if (node->output) {
            node->output->ir_node = wmma;
            node->output->ir_context = ir;
        }
    }

    unlink_node(ir, node);
    free_node(node);

    LOG_DEBUG("TC opt: rewrote MATMUL [%d,%d,%d] -> WMMA [%d,%d,%d]", m, n, k, pm, pn, pk);
    return 1;
}

static bool is_fused_matmul_pattern(CMLGraph_t ir, struct IRNode* reduce_node,
                                    struct IRNode** out_a, struct IRNode** out_b) {
    if (!reduce_node || reduce_node->type != UOP_SUM || reduce_node->num_inputs != 1)
        return false;

    struct IRNode* mul = find_node_by_output(ir, reduce_node->input_names[0]);
    if (!mul || mul->type != UOP_MUL || mul->num_inputs != 2)
        return false;

    struct IRNode* expand_a = find_node_by_output(ir, mul->input_names[0]);
    struct IRNode* expand_b = find_node_by_output(ir, mul->input_names[1]);

    if (!expand_a || expand_a->type != UOP_EXPAND || expand_a->num_inputs != 1)
        return false;
    if (!expand_b || expand_b->type != UOP_EXPAND || expand_b->num_inputs != 1)
        return false;

    *out_a = find_node_by_output(ir, expand_a->input_names[0]);
    *out_b = find_node_by_output(ir, expand_b->input_names[0]);

    if (!*out_a || !*out_b) return false;

    if (!(*out_a)->output_shape || !(*out_b)->output_shape) return false;
    if ((*out_a)->output_ndim < 2 || (*out_b)->output_ndim < 2) return false;

    return true;
}

static int rewrite_fused_matmul(CMLGraph_t ir, struct IRNode* reduce_node,
                                struct IRNode* src_a, struct IRNode* src_b) {
    int ndim_a = src_a->output_ndim;
    int ndim_b = src_b->output_ndim;
    int m = src_a->output_shape[ndim_a - 2];
    int k = src_a->output_shape[ndim_a - 1];
    int n = src_b->output_shape[ndim_b - 1];

    if (!dims_tc_compatible(m, n, k, &g_tc_config))
        return 0;

    int pm = round_up(m, 16);
    int pn = round_up(n, 16);
    int pk = round_up(k, 16);

    int out_ndim = reduce_node->output_ndim > 0 ? reduce_node->output_ndim : 2;
    int* out_shape = malloc((size_t)out_ndim * sizeof(int));
    if (!out_shape) return 0;

    if (reduce_node->output_shape && reduce_node->output_ndim > 0)
        memcpy(out_shape, reduce_node->output_shape, (size_t)out_ndim * sizeof(int));
    else {
        out_shape[0] = pm;
        out_shape[1] = pn;
    }

    struct IRNode* wmma = create_wmma_node(
        src_a->output_name, src_b->output_name,
        pm, pn, pk, out_shape, out_ndim);
    free(out_shape);
    if (!wmma) return 0;

    insert_before(ir, wmma, reduce_node);

    if (reduce_node->output_name)
        replace_refs(ir, reduce_node->output_name, wmma->output_name);
    if (reduce_node->output) {
        reduce_node->output->ir_node = wmma;
        reduce_node->output->ir_context = ir;
    }

    LOG_DEBUG("TC opt: rewrote SUM(MUL(EXPAND,EXPAND)) -> WMMA [%d,%d,%d]", m, n, k);
    return 1;
}

int cml_tc_optimize(CMLGraph_t graph) {
    if (!graph || !graph->head) return 0;
    if (!cml_tc_available()) return 0;

    CMLTCHardware hw = tc_detect_hardware();
    int tile_m, tile_n, tile_k;
    tc_get_tile_size(hw, &tile_m, &tile_n, &tile_k);
    (void)tile_m; (void)tile_n; (void)tile_k;

    int rewrites = 0;
    struct IRNode* node = graph->head;

    while (node) {
        struct IRNode* next = node->next;

        if (node->type == UOP_MATMUL && !node->is_fused) {
            rewrites += rewrite_matmul_to_wmma(graph, node);
        } else if (node->type == UOP_SUM) {
            struct IRNode* a = NULL;
            struct IRNode* b = NULL;
            if (is_fused_matmul_pattern(graph, node, &a, &b))
                rewrites += rewrite_fused_matmul(graph, node, a, b);
        }

        node = next;
    }

    if (rewrites > 0)
        LOG_INFO("TC optimization: %d matmul(s) rewritten to use tensor cores", rewrites);

    return rewrites;
}
