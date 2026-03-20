#include "ops/ir/rangeify.h"
#include "ops/ir/internal.h"
#include "ops/ir/schedule.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

RangeProgram* range_program_create(void) {
    RangeProgram* prog = calloc(1, sizeof(RangeProgram));
    if (!prog) return NULL;
    prog->next_id = 0;
    return prog;
}

void range_program_free(RangeProgram* prog) {
    if (!prog) return;
    RangeNode* cur = prog->head;
    while (cur) {
        RangeNode* next = cur->next;
        free(cur);
        cur = next;
    }
    free(prog);
}

static RangeNode* alloc_node(RangeProgram* prog, RangeUOpType type) {
    RangeNode* node = calloc(1, sizeof(RangeNode));
    if (!node) return NULL;
    node->type = type;
    node->id = prog->next_id++;
    node->next = NULL;
    if (!prog->head) {
        prog->head = node;
        prog->tail = node;
    } else {
        prog->tail->next = node;
        prog->tail = node;
    }
    prog->num_nodes++;
    return node;
}

int range_program_add_range(RangeProgram* prog, int start, int end, int step, int dim) {
    if (!prog || end <= start || step <= 0) return -1;
    RangeNode* node = alloc_node(prog, UOP_RANGE);
    if (!node) return -1;
    node->range.start = start;
    node->range.end = end;
    node->range.step = step;
    node->range.dim = dim;
    return node->id;
}

int range_program_add_index(RangeProgram* prog, int* range_ids, size_t* strides, int num_ranges) {
    if (!prog || !range_ids || !strides || num_ranges <= 0 || num_ranges > 8)
        return -1;
    RangeNode* node = alloc_node(prog, UOP_INDEX);
    if (!node) return -1;
    node->index.num_ranges = num_ranges;
    memcpy(node->index.range_ids, range_ids, (size_t)num_ranges * sizeof(int));
    memcpy(node->index.strides, strides, (size_t)num_ranges * sizeof(size_t));
    return node->id;
}

void range_program_print(const RangeProgram* prog) {
    if (!prog) {
        printf("RangeProgram: (null)\n");
        return;
    }
    printf("RangeProgram (%d nodes)\n", prog->num_nodes);
    for (RangeNode* n = prog->head; n; n = n->next) {
        if (n->type == UOP_RANGE) {
            printf("  [%d] RANGE  dim=%d start=%d end=%d step=%d\n",
                   n->id, n->range.dim, n->range.start, n->range.end, n->range.step);
        } else {
            printf("  [%d] INDEX  ranges=[", n->id);
            for (int i = 0; i < n->index.num_ranges; i++) {
                if (i > 0) printf(", ");
                printf("r%d*%zu", n->index.range_ids[i], n->index.strides[i]);
            }
            printf("]\n");
        }
    }
}

static size_t* compute_strides_for_shape(const int* shape, int ndim) {
    if (!shape || ndim <= 0) return NULL;
    size_t* strides = malloc((size_t)ndim * sizeof(size_t));
    if (!strides) return NULL;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        strides[i] = strides[i + 1] * (size_t)shape[i + 1];
    return strides;
}

static RangeProgram* rangeify_node(struct IRNode* node) {
    if (!node || !node->output_shape || node->output_ndim <= 0)
        return NULL;

    int ndim = node->output_ndim;
    int* shape = node->output_shape;

    RangeProgram* prog = range_program_create();
    if (!prog) return NULL;

    int range_ids[8];
    if (ndim > 8) ndim = 8;

    for (int d = 0; d < ndim; d++) {
        int rid = range_program_add_range(prog, 0, shape[d], 1, d);
        if (rid < 0) {
            range_program_free(prog);
            return NULL;
        }
        range_ids[d] = rid;
    }

    size_t* out_strides = compute_strides_for_shape(shape, ndim);
    if (!out_strides) {
        range_program_free(prog);
        return NULL;
    }
    range_program_add_index(prog, range_ids, out_strides, ndim);

    for (int i = 0; i < node->num_inputs; i++) {
        Tensor* inp = (node->inputs) ? node->inputs[i] : NULL;
        if (!inp || !inp->shape || inp->ndim <= 0) continue;

        size_t* inp_strides = compute_strides_for_shape(inp->shape, inp->ndim);
        if (!inp_strides) continue;

        int inp_ndim = inp->ndim;
        if (inp_ndim > 8) inp_ndim = 8;

        int offset = ndim - inp_ndim;
        if (offset < 0) offset = 0;

        int inp_range_ids[8];
        size_t inp_effective_strides[8];
        int count = 0;

        for (int d = 0; d < inp_ndim && count < 8; d++) {
            int out_d = d + offset;
            if (out_d >= ndim) break;
            if (inp->shape[d] == 1 && shape[out_d] > 1) {
                inp_range_ids[count] = range_ids[out_d];
                inp_effective_strides[count] = 0;
            } else {
                inp_range_ids[count] = range_ids[out_d];
                inp_effective_strides[count] = inp_strides[d];
            }
            count++;
        }

        if (count > 0)
            range_program_add_index(prog, inp_range_ids, inp_effective_strides, count);
        free(inp_strides);
    }

    free(out_strides);
    return prog;
}

int cml_rangeify(CMLGraph_t graph) {
    if (!graph) return -1;

    struct CMLGraph* g = (struct CMLGraph*)graph;
    int converted = 0;

    for (struct IRNode* node = g->head; node; node = node->next) {
        if (!node->output_shape || node->output_ndim <= 0)
            continue;

        bool is_elem = cml_schedule_is_elementwise(node->type);
        bool is_reduce = cml_schedule_is_reduction(node->type);
        if (!is_elem && !is_reduce)
            continue;

        RangeProgram* rp = rangeify_node(node);
        if (!rp) continue;

        LOG_DEBUG("Rangeify: converted %s node (%d dims -> %d range nodes)",
                  uop_type_to_string(node->type), node->output_ndim, rp->num_nodes);

        range_program_free(rp);
        converted++;
    }

    LOG_DEBUG("Rangeify pass: converted %d nodes", converted);
    return converted;
}
