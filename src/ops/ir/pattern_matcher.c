/**
 * @file pattern_matcher.c
 * @brief Declarative pattern-matcher and rewriter for IR graphs
 *
 * Matches IRNode subgraphs against declarative patterns and applies
 * rewrite rules (algebraic simplifications, canonicalisations, etc.).
 */

#include "ops/ir/pattern_matcher.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>

/* ================================================================
 * Internal helpers
 * ================================================================ */

/** Counter for generating unique output names for replacement nodes. */
static atomic_int g_rewrite_counter = 0;

/** Generate a unique name prefixed with "_rw" for rewrite-emitted nodes. */
static char* rewrite_unique_name(void) {
    int id = atomic_fetch_add(&g_rewrite_counter, 1);
    char* name = malloc(32);
    if (name) {
        snprintf(name, 32, "_rw%d", id);
    }
    return name;
}

/**
 * Find the IR node whose output_name matches @p name.
 * Linear scan over the graph's linked list.
 */
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

/**
 * Record a capture in a CMLMatchResult.
 * Returns 0 on success, -1 if the capture table is full.
 */
static int record_capture(CMLMatchResult* result, const char* name, struct IRNode* node) {
    if (!result || !name || !node) return -1;
    if (result->num_captures >= CML_PATTERN_MAX_CAPTURES) return -1;

    /* Check for duplicate capture name -- if already captured, the node must match. */
    for (int i = 0; i < result->num_captures; i++) {
        if (strcmp(result->captures[i].name, name) == 0) {
            return (result->captures[i].node == node) ? 0 : -1;
        }
    }

    CMLCaptureEntry* e = &result->captures[result->num_captures];
    strncpy(e->name, name, sizeof(e->name) - 1);
    e->name[sizeof(e->name) - 1] = '\0';
    e->node = node;
    result->num_captures++;
    return 0;
}

/**
 * Retrieve a previously captured node by name.
 * Returns NULL if not found.
 *
 * This helper is available for user-defined emit functions that look up
 * captures by name.  The built-in rules do direct lookups instead, so
 * this function may appear unused in the built-in set.
 */
__attribute__((unused))
static struct IRNode* get_capture(const CMLMatchResult* result, const char* name) {
    if (!result || !name) return NULL;
    for (int i = 0; i < result->num_captures; i++) {
        if (strcmp(result->captures[i].name, name) == 0)
            return result->captures[i].node;
    }
    return NULL;
}

/* ================================================================
 * Core matching engine
 * ================================================================ */

/**
 * Try to match @p pattern against @p node within the graph @p ir.
 * Fills @p result with any captures.
 *
 * @return true if pattern matches, false otherwise.
 */
static bool match_node(CMLGraph_t ir, const CMLPatternNode* pattern,
                       struct IRNode* node, CMLMatchResult* result) {
    if (!pattern || !node) return false;

    switch (pattern->kind) {
    case CML_PAT_ANY:
        /* Wildcard -- always matches any node. */
        return true;

    case CML_PAT_CAPTURE:
        /* Capture -- matches any node and records it by name. */
        return record_capture(result, pattern->capture_name, node) == 0;

    case CML_PAT_OP: {
        /* Must match the UOpType. */
        if (node->type != pattern->op_type)
            return false;

        /* Number of pattern inputs must match the node's input count. */
        if (pattern->num_inputs != node->num_inputs)
            return false;

        /* Recursively match each input sub-pattern against the
         * producer of the corresponding input_name in the graph. */
        for (int i = 0; i < pattern->num_inputs; i++) {
            if (!pattern->inputs[i])
                return false;

            /* Find the producer node for this input. */
            struct IRNode* producer = find_node_by_output(ir, node->input_names[i]);
            if (!producer)
                return false;

            if (!match_node(ir, pattern->inputs[i], producer, result))
                return false;
        }
        return true;
    }

    default:
        return false;
    }
}

/* ================================================================
 * Replacement helpers
 * ================================================================ */

/**
 * After a rewrite produces a replacement node, rewire all downstream
 * nodes that referenced the old output_name to use the new output_name.
 */
static void replace_output_references(CMLGraph_t ir,
                                      const char* old_name,
                                      const char* new_name) {
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

/**
 * Insert @p new_node into the graph immediately before @p before.
 * If @p before is NULL the node is appended at the tail.
 */
static void insert_node_before(CMLGraph_t ir, struct IRNode* new_node,
                               struct IRNode* before) {
    if (!ir || !new_node) return;

    new_node->next = NULL;

    if (!before || !ir->head) {
        /* Append at tail. */
        if (ir->tail) {
            ir->tail->next = new_node;
        } else {
            ir->head = new_node;
        }
        ir->tail = new_node;
        ir->node_count++;
        return;
    }

    /* Find predecessor of 'before'. */
    if (ir->head == before) {
        new_node->next = before;
        ir->head = new_node;
        ir->node_count++;
        return;
    }

    struct IRNode* prev = ir->head;
    while (prev && prev->next != before) {
        prev = prev->next;
    }

    if (prev) {
        new_node->next = before;
        prev->next = new_node;
        ir->node_count++;
    } else {
        /* 'before' not found in list -- append at tail. */
        ir->tail->next = new_node;
        ir->tail = new_node;
        ir->node_count++;
    }
}

/**
 * Remove @p node from the linked list (does NOT free it).
 */
static void unlink_node(CMLGraph_t ir, struct IRNode* node) {
    if (!ir || !node) return;

    if (ir->head == node) {
        ir->head = node->next;
        if (ir->tail == node) ir->tail = NULL;
        ir->node_count--;
        return;
    }

    struct IRNode* prev = ir->head;
    while (prev && prev->next != node) {
        prev = prev->next;
    }
    if (prev) {
        prev->next = node->next;
        if (ir->tail == node) ir->tail = prev;
        ir->node_count--;
    }
}

/**
 * Free an IRNode that has been unlinked from the graph.
 * Only frees the node-level bookkeeping (names, etc.), not
 * deep resources like tensor data or broadcast info.
 */
static void free_unlinked_node(struct IRNode* node) {
    if (!node) return;

    if (node->input_names) {
        for (int i = 0; i < node->num_inputs; i++) {
            free(node->input_names[i]);
        }
        free(node->input_names);
    }
    free(node->output_name);
    free(node->users);

    /* Clear tensor's back-pointer to avoid dangling refs. */
    if (node->output) {
        node->output->ir_node = NULL;
        node->output->ir_context = NULL;
    }

    free(node);
}

/* ================================================================
 * Pattern builder helpers (public API)
 * ================================================================ */

CMLPatternNode* cml_pattern_op(UOpType type, CMLPatternNode** inputs, int num_inputs) {
    if (num_inputs < 0 || num_inputs > CML_PATTERN_MAX_INPUTS) return NULL;

    CMLPatternNode* p = calloc(1, sizeof(CMLPatternNode));
    if (!p) return NULL;

    p->kind = CML_PAT_OP;
    p->op_type = type;
    p->num_inputs = num_inputs;

    for (int i = 0; i < num_inputs; i++) {
        p->inputs[i] = inputs ? inputs[i] : NULL;
    }
    return p;
}

CMLPatternNode* cml_pattern_capture(const char* name) {
    if (!name) return NULL;

    CMLPatternNode* p = calloc(1, sizeof(CMLPatternNode));
    if (!p) return NULL;

    p->kind = CML_PAT_CAPTURE;
    strncpy(p->capture_name, name, sizeof(p->capture_name) - 1);
    p->capture_name[sizeof(p->capture_name) - 1] = '\0';
    p->num_inputs = 0;
    return p;
}

CMLPatternNode* cml_pattern_any(void) {
    CMLPatternNode* p = calloc(1, sizeof(CMLPatternNode));
    if (!p) return NULL;

    p->kind = CML_PAT_ANY;
    p->num_inputs = 0;
    return p;
}

void cml_pattern_free(CMLPatternNode* node) {
    if (!node) return;

    for (int i = 0; i < node->num_inputs; i++) {
        cml_pattern_free(node->inputs[i]);
    }
    free(node);
}

/* ================================================================
 * Registry API (public)
 * ================================================================ */

CMLRewriteRegistry* cml_rewrite_registry_create(void) {
    CMLRewriteRegistry* reg = calloc(1, sizeof(CMLRewriteRegistry));
    return reg; /* num_rules is 0 thanks to calloc */
}

void cml_rewrite_registry_free(CMLRewriteRegistry* reg) {
    if (!reg) return;

    for (int i = 0; i < reg->num_rules; i++) {
        cml_pattern_free(reg->rules[i].pattern);
        /* emit function pointer and name string literal are not owned */
    }
    free(reg);
}

int cml_rewrite_register(CMLRewriteRegistry* reg, CMLPatternNode* pattern,
                         CMLEmitFn emit, int priority, const char* name) {
    if (!reg || !pattern) return -1;
    if (reg->num_rules >= CML_REWRITE_MAX_RULES) {
        LOG_ERROR("Rewrite registry full (%d rules)", CML_REWRITE_MAX_RULES);
        return -1;
    }

    /* Find insertion index to keep rules sorted by descending priority. */
    int idx = reg->num_rules;
    for (int i = 0; i < reg->num_rules; i++) {
        if (priority > reg->rules[i].priority) {
            idx = i;
            break;
        }
    }

    /* Shift rules to make room. */
    if (idx < reg->num_rules) {
        memmove(&reg->rules[idx + 1], &reg->rules[idx],
                (size_t)(reg->num_rules - idx) * sizeof(CMLRewriteRule));
    }

    reg->rules[idx].pattern  = pattern;
    reg->rules[idx].emit     = emit;
    reg->rules[idx].priority = priority;
    reg->rules[idx].name     = name;
    reg->num_rules++;

    LOG_DEBUG("Registered rewrite rule '%s' (priority %d, slot %d)",
              name ? name : "<unnamed>", priority, idx);
    return 0;
}

/* ================================================================
 * cml_rewrite_apply  -- main rewrite loop
 * ================================================================ */

int cml_rewrite_apply(CMLRewriteRegistry* reg, CMLGraph_t ir, int max_iterations) {
    if (!reg || !ir) return -1;
    if (reg->num_rules == 0) return 0;

    int max_iter = (max_iterations > 0) ? max_iterations : CML_REWRITE_DEFAULT_MAX_ITER;
    int total_rewrites = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        int rewrites_this_pass = 0;

        struct IRNode* node = ir->head;
        while (node) {
            struct IRNode* next_node = node->next; /* save in case node is removed */
            bool matched = false;

            /* Try every rule (already sorted by descending priority). */
            for (int r = 0; r < reg->num_rules && !matched; r++) {
                CMLRewriteRule* rule = &reg->rules[r];

                /* Prepare match result. */
                CMLMatchResult result;
                memset(&result, 0, sizeof(result));
                result.matched_root = node;

                if (match_node(ir, rule->pattern, node, &result)) {
                    /* Pattern matched -- invoke emit to produce a replacement. */
                    struct IRNode* replacement = rule->emit(ir, &result);

                    if (replacement && replacement != node) {
                        /* Give the replacement a unique output name if it
                         * does not already have one. */
                        if (!replacement->output_name) {
                            replacement->output_name = rewrite_unique_name();
                        }

                        /* If the replacement node is not already in the
                         * graph, insert it just before the matched node. */
                        bool already_in_graph = false;
                        {
                            struct IRNode* scan = ir->head;
                            while (scan) {
                                if (scan == replacement) { already_in_graph = true; break; }
                                scan = scan->next;
                            }
                        }
                        if (!already_in_graph) {
                            insert_node_before(ir, replacement, node);
                        }

                        /* Rewire all downstream references from the old
                         * node's output to the replacement's output. */
                        if (node->output_name && replacement->output_name) {
                            replace_output_references(ir, node->output_name,
                                                      replacement->output_name);
                        }

                        /* If the old node had an output tensor, point it to
                         * the replacement so lazy evaluation still works. */
                        if (node->output && replacement->output) {
                            node->output->ir_node    = replacement;
                            node->output->ir_context = ir;
                        }

                        /* Unlink and free the old node. */
                        unlink_node(ir, node);
                        free_unlinked_node(node);

                        rewrites_this_pass++;
                        matched = true;

                        LOG_DEBUG("Rewrite '%s' applied (iter %d)",
                                  rule->name ? rule->name : "?", iter);
                    } else if (replacement == node) {
                        /* Emit returned the same node -- no-op (pattern
                         * matched but replacement is identity). */
                    }
                }
            }

            node = next_node;
        }

        total_rewrites += rewrites_this_pass;

        if (rewrites_this_pass == 0) {
            LOG_DEBUG("Rewrite converged after %d iteration(s), %d total rewrites",
                      iter + 1, total_rewrites);
            break;
        }
    }

    return total_rewrites;
}

/* ================================================================
 * Built-in algebraic simplification rules
 * ================================================================ */

/* ---- Helpers for checking FILL constants ---- */

/**
 * Return true if @p node is a UOP_FILL whose constant value equals @p value.
 */
static bool is_fill_const(struct IRNode* node, float value) {
    if (!node || node->type != UOP_FILL || !node->params)
        return false;
    FillParams* fp = (FillParams*)node->params;
    /* Use a small epsilon for float comparison. */
    float diff = fp->value - value;
    if (diff < 0) diff = -diff;
    return diff < 1e-7f;
}

/* ---- Helper: create a FILL node for a scalar constant ---- */

static struct IRNode* make_fill_node(CMLGraph_t ir, float value,
                                     struct IRNode* shape_donor) {
    (void)ir;
    struct IRNode* node = calloc(1, sizeof(struct IRNode));
    if (!node) return NULL;

    node->type = UOP_FILL;
    node->num_inputs = 0;
    node->input_names = NULL;
    node->output_name = rewrite_unique_name();
    node->next = NULL;

    FillParams* fp = malloc(sizeof(FillParams));
    if (!fp) { free(node->output_name); free(node); return NULL; }

    fp->value = value;

    /* Copy shape from the shape_donor if available. */
    if (shape_donor && shape_donor->output_shape && shape_donor->output_ndim > 0) {
        fp->ndim = shape_donor->output_ndim;
        fp->shape = malloc((size_t)fp->ndim * sizeof(int));
        if (fp->shape) {
            memcpy(fp->shape, shape_donor->output_shape,
                   (size_t)fp->ndim * sizeof(int));
        }
        node->output_shape = malloc((size_t)fp->ndim * sizeof(int));
        if (node->output_shape) {
            memcpy(node->output_shape, shape_donor->output_shape,
                   (size_t)fp->ndim * sizeof(int));
        }
        node->output_ndim = fp->ndim;
    } else {
        /* Scalar: shape = {1} */
        fp->ndim = 1;
        fp->shape = malloc(sizeof(int));
        if (fp->shape) fp->shape[0] = 1;
        node->output_ndim = 1;
        node->output_shape = malloc(sizeof(int));
        if (node->output_shape) node->output_shape[0] = 1;
    }

    node->params = fp;
    return node;
}

/* ---- Rule 1: x * 1 -> x ---- */

static struct IRNode* emit_mul_one(CMLGraph_t ir, const CMLMatchResult* m) {
    (void)ir;
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_MUL || root->num_inputs != 2) return NULL;

    /* Find producer of each operand. */
    struct IRNode* lhs = find_node_by_output(ir, root->input_names[0]);
    struct IRNode* rhs = find_node_by_output(ir, root->input_names[1]);
    if (!lhs || !rhs) return NULL;

    if (is_fill_const(rhs, 1.0f)) return lhs;
    if (is_fill_const(lhs, 1.0f)) return rhs;
    return NULL;
}

/* ---- Rule 2: x + 0 -> x ---- */

static struct IRNode* emit_add_zero(CMLGraph_t ir, const CMLMatchResult* m) {
    (void)ir;
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_ADD || root->num_inputs != 2) return NULL;

    struct IRNode* lhs = find_node_by_output(ir, root->input_names[0]);
    struct IRNode* rhs = find_node_by_output(ir, root->input_names[1]);
    if (!lhs || !rhs) return NULL;

    if (is_fill_const(rhs, 0.0f)) return lhs;
    if (is_fill_const(lhs, 0.0f)) return rhs;
    return NULL;
}

/* ---- Rule 3: x - x -> 0 ---- */

static struct IRNode* emit_sub_self(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_SUB || root->num_inputs != 2) return NULL;

    /* Both inputs reference the same producer. */
    if (!root->input_names[0] || !root->input_names[1]) return NULL;
    if (strcmp(root->input_names[0], root->input_names[1]) != 0) return NULL;

    return make_fill_node(ir, 0.0f, root);
}

/* ---- Rule 4: exp(log(x)) -> x ---- */

static struct IRNode* emit_exp_log(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_EXP || root->num_inputs != 1) return NULL;

    struct IRNode* inner = find_node_by_output(ir, root->input_names[0]);
    if (!inner || inner->type != UOP_LOG || inner->num_inputs != 1) return NULL;

    /* Return the input to LOG -- that is, x. */
    struct IRNode* x = find_node_by_output(ir, inner->input_names[0]);
    return x;
}

/* ---- Rule 5: neg(neg(x)) -> x ---- */

static struct IRNode* emit_neg_neg(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_NEG || root->num_inputs != 1) return NULL;

    struct IRNode* inner = find_node_by_output(ir, root->input_names[0]);
    if (!inner || inner->type != UOP_NEG || inner->num_inputs != 1) return NULL;

    struct IRNode* x = find_node_by_output(ir, inner->input_names[0]);
    return x;
}

/* ---- Rule 6: relu6(relu6(x)) -> relu6(x) ---- */
/*
 * Note: The codebase has no UOP_RELU enum value; ReLU is decomposed into
 * max(x, 0) using UOP_MAX.  UOP_RELU6 is the closest activation op in the
 * enum, so the idempotent rule is expressed for RELU6 instead.
 */

static struct IRNode* emit_relu6_relu6(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_RELU6 || root->num_inputs != 1) return NULL;

    struct IRNode* inner = find_node_by_output(ir, root->input_names[0]);
    if (!inner || inner->type != UOP_RELU6) return NULL;

    /* relu6(relu6(x)) == relu6(x) -- return the inner node directly. */
    return inner;
}

/* ---- Constant folding helpers ---- */

/** Extract the fill value from a FILL node, return false if not FILL */
static bool get_fill_value(struct IRNode* node, float* out) {
    if (!node || node->type != UOP_FILL || !node->params) return false;
    FillParams* fp = (FillParams*)node->params;
    *out = fp->value;
    return true;
}

/* ---- Rule 7: FILL(a) + FILL(b) -> FILL(a+b) ---- */
static struct IRNode* emit_fold_add(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_ADD || root->num_inputs != 2) return NULL;
    struct IRNode* lhs = find_node_by_output(ir, root->input_names[0]);
    struct IRNode* rhs = find_node_by_output(ir, root->input_names[1]);
    float a, b;
    if (!get_fill_value(lhs, &a) || !get_fill_value(rhs, &b)) return NULL;
    return make_fill_node(ir, a + b, root);
}

/* ---- Rule 8: FILL(a) - FILL(b) -> FILL(a-b) ---- */
static struct IRNode* emit_fold_sub(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_SUB || root->num_inputs != 2) return NULL;
    struct IRNode* lhs = find_node_by_output(ir, root->input_names[0]);
    struct IRNode* rhs = find_node_by_output(ir, root->input_names[1]);
    float a, b;
    if (!get_fill_value(lhs, &a) || !get_fill_value(rhs, &b)) return NULL;
    return make_fill_node(ir, a - b, root);
}

/* ---- Rule 9: FILL(a) * FILL(b) -> FILL(a*b) ---- */
static struct IRNode* emit_fold_mul(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_MUL || root->num_inputs != 2) return NULL;
    struct IRNode* lhs = find_node_by_output(ir, root->input_names[0]);
    struct IRNode* rhs = find_node_by_output(ir, root->input_names[1]);
    float a, b;
    if (!get_fill_value(lhs, &a) || !get_fill_value(rhs, &b)) return NULL;
    return make_fill_node(ir, a * b, root);
}

/* ---- Rule 10: FILL(a) / FILL(b) -> FILL(a/b) ---- */
static struct IRNode* emit_fold_div(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_DIV || root->num_inputs != 2) return NULL;
    struct IRNode* lhs = find_node_by_output(ir, root->input_names[0]);
    struct IRNode* rhs = find_node_by_output(ir, root->input_names[1]);
    float a, b;
    if (!get_fill_value(lhs, &a) || !get_fill_value(rhs, &b)) return NULL;
    if (b == 0.0f) return NULL; /* avoid div-by-zero */
    return make_fill_node(ir, a / b, root);
}

/* ---- Rule 11: NEG(FILL(a)) -> FILL(-a) ---- */
static struct IRNode* emit_fold_neg(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_NEG || root->num_inputs != 1) return NULL;
    struct IRNode* inner = find_node_by_output(ir, root->input_names[0]);
    float a;
    if (!get_fill_value(inner, &a)) return NULL;
    return make_fill_node(ir, -a, root);
}

/* ---- Rule 12: x * 2 -> x + x ---- */
static struct IRNode* emit_mul_two(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_MUL || root->num_inputs != 2) return NULL;
    struct IRNode* lhs = find_node_by_output(ir, root->input_names[0]);
    struct IRNode* rhs = find_node_by_output(ir, root->input_names[1]);
    if (!lhs || !rhs) return NULL;

    /* Check if either operand is FILL(2) */
    const char* x_name = NULL;
    if (is_fill_const(rhs, 2.0f)) x_name = root->input_names[0];
    else if (is_fill_const(lhs, 2.0f)) x_name = root->input_names[1];
    if (!x_name) return NULL;

    /* Create ADD(x, x) */
    struct IRNode* node = calloc(1, sizeof(struct IRNode));
    if (!node) return NULL;
    node->type = UOP_ADD;
    node->num_inputs = 2;
    node->input_names = malloc(2 * sizeof(char*));
    if (!node->input_names) { free(node); return NULL; }
    node->input_names[0] = strdup(x_name);
    node->input_names[1] = strdup(x_name);
    node->output_name = rewrite_unique_name();
    node->output_ndim = root->output_ndim;
    if (root->output_shape && root->output_ndim > 0) {
        node->output_shape = malloc((size_t)root->output_ndim * sizeof(int));
        if (node->output_shape)
            memcpy(node->output_shape, root->output_shape,
                   (size_t)root->output_ndim * sizeof(int));
    }
    return node;
}

/* ---- Rule 13: x * 0 -> FILL(0) ---- */
static struct IRNode* emit_mul_zero(CMLGraph_t ir, const CMLMatchResult* m) {
    (void)ir;
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_MUL || root->num_inputs != 2) return NULL;
    struct IRNode* lhs = find_node_by_output(ir, root->input_names[0]);
    struct IRNode* rhs = find_node_by_output(ir, root->input_names[1]);
    if (!lhs || !rhs) return NULL;
    if (!is_fill_const(rhs, 0.0f) && !is_fill_const(lhs, 0.0f)) return NULL;
    return make_fill_node(ir, 0.0f, root);
}

/* ---- Rule 14: x / const -> x * (1/const) ---- */
static struct IRNode* emit_div_const(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_DIV || root->num_inputs != 2) return NULL;
    struct IRNode* rhs = find_node_by_output(ir, root->input_names[1]);
    float b;
    if (!get_fill_value(rhs, &b)) return NULL;
    if (b == 0.0f || b == 1.0f) return NULL; /* skip trivial or unsafe */

    /* Create FILL(1/b) node */
    struct IRNode* recip_node = make_fill_node(ir, 1.0f / b, root);
    if (!recip_node) return NULL;

    /* Create MUL(x, recip) node */
    insert_node_before(ir, recip_node, root);

    struct IRNode* mul_node = calloc(1, sizeof(struct IRNode));
    if (!mul_node) return NULL;
    mul_node->type = UOP_MUL;
    mul_node->num_inputs = 2;
    mul_node->input_names = malloc(2 * sizeof(char*));
    if (!mul_node->input_names) { free(mul_node); return NULL; }
    mul_node->input_names[0] = strdup(root->input_names[0]);
    mul_node->input_names[1] = strdup(recip_node->output_name);
    mul_node->output_name = rewrite_unique_name();
    mul_node->output_ndim = root->output_ndim;
    if (root->output_shape && root->output_ndim > 0) {
        mul_node->output_shape = malloc((size_t)root->output_ndim * sizeof(int));
        if (mul_node->output_shape)
            memcpy(mul_node->output_shape, root->output_shape,
                   (size_t)root->output_ndim * sizeof(int));
    }
    return mul_node;
}

/* ---- Rule 15: log(exp(x)) -> x ---- */
static struct IRNode* emit_log_exp(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_LOG || root->num_inputs != 1) return NULL;
    struct IRNode* inner = find_node_by_output(ir, root->input_names[0]);
    if (!inner || inner->type != UOP_EXP || inner->num_inputs != 1) return NULL;
    return find_node_by_output(ir, inner->input_names[0]);
}

/* ---- Rule 16: sqrt(x) * sqrt(x) -> x ---- */
static struct IRNode* emit_sqrt_sq(CMLGraph_t ir, const CMLMatchResult* m) {
    struct IRNode* root = m->matched_root;
    if (!root || root->type != UOP_MUL || root->num_inputs != 2) return NULL;
    /* Both operands must be the same SQRT node */
    if (!root->input_names[0] || !root->input_names[1]) return NULL;
    if (strcmp(root->input_names[0], root->input_names[1]) != 0) return NULL;
    struct IRNode* sq = find_node_by_output(ir, root->input_names[0]);
    if (!sq || sq->type != UOP_SQRT || sq->num_inputs != 1) return NULL;
    return find_node_by_output(ir, sq->input_names[0]);
}

/* ---- Patterns for the built-in rules ---- */

/**
 * Build pattern: OP(capture("a"), capture("b"))
 * Used for MUL, ADD, SUB patterns with 2 inputs.
 */
static CMLPatternNode* make_binop_pattern(UOpType type) {
    CMLPatternNode* a = cml_pattern_capture("a");
    CMLPatternNode* b = cml_pattern_capture("b");
    if (!a || !b) { cml_pattern_free(a); cml_pattern_free(b); return NULL; }
    CMLPatternNode* inputs[2] = { a, b };
    return cml_pattern_op(type, inputs, 2);
}

/**
 * Build pattern: OUTER(INNER(capture("x")))
 * Used for exp(log(x)), neg(neg(x)), relu6(relu6(x)).
 */
static CMLPatternNode* make_unary_chain_pattern(UOpType outer, UOpType inner) {
    CMLPatternNode* x = cml_pattern_capture("x");
    if (!x) return NULL;
    CMLPatternNode* inner_inputs[1] = { x };
    CMLPatternNode* inner_node = cml_pattern_op(inner, inner_inputs, 1);
    if (!inner_node) { cml_pattern_free(x); return NULL; }
    CMLPatternNode* outer_inputs[1] = { inner_node };
    return cml_pattern_op(outer, outer_inputs, 1);
}

/* ---- Public: create registry with all built-in rules ---- */

CMLRewriteRegistry* cml_rewrite_builtin_rules(void) {
    CMLRewriteRegistry* reg = cml_rewrite_registry_create();
    if (!reg) return NULL;

    /* Rule 1: x * 1 -> x  (priority 100) */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_MUL);
        if (pat) cml_rewrite_register(reg, pat, emit_mul_one, 100, "mul_one");
    }

    /* Rule 2: x + 0 -> x  (priority 100) */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_ADD);
        if (pat) cml_rewrite_register(reg, pat, emit_add_zero, 100, "add_zero");
    }

    /* Rule 3: x - x -> 0  (priority 90) */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_SUB);
        if (pat) cml_rewrite_register(reg, pat, emit_sub_self, 90, "sub_self");
    }

    /* Rule 4: exp(log(x)) -> x  (priority 80) */
    {
        CMLPatternNode* pat = make_unary_chain_pattern(UOP_EXP, UOP_LOG);
        if (pat) cml_rewrite_register(reg, pat, emit_exp_log, 80, "exp_log");
    }

    /* Rule 5: neg(neg(x)) -> x  (priority 80) */
    {
        CMLPatternNode* pat = make_unary_chain_pattern(UOP_NEG, UOP_NEG);
        if (pat) cml_rewrite_register(reg, pat, emit_neg_neg, 80, "neg_neg");
    }

    /* Rule 6: relu6(relu6(x)) -> relu6(x)  (priority 70) */
    {
        CMLPatternNode* pat = make_unary_chain_pattern(UOP_RELU6, UOP_RELU6);
        if (pat) cml_rewrite_register(reg, pat, emit_relu6_relu6, 70, "relu6_relu6");
    }

    /* ── Constant folding rules (priority 120) ── */

    /* Rule 7: FILL(a) + FILL(b) -> FILL(a+b) */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_ADD);
        if (pat) cml_rewrite_register(reg, pat, emit_fold_add, 120, "fold_add");
    }

    /* Rule 8: FILL(a) - FILL(b) -> FILL(a-b) */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_SUB);
        if (pat) cml_rewrite_register(reg, pat, emit_fold_sub, 120, "fold_sub");
    }

    /* Rule 9: FILL(a) * FILL(b) -> FILL(a*b) */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_MUL);
        if (pat) cml_rewrite_register(reg, pat, emit_fold_mul, 120, "fold_mul");
    }

    /* Rule 10: FILL(a) / FILL(b) -> FILL(a/b) */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_DIV);
        if (pat) cml_rewrite_register(reg, pat, emit_fold_div, 120, "fold_div");
    }

    /* Rule 11: NEG(FILL(a)) -> FILL(-a) */
    {
        CMLPatternNode* x = cml_pattern_capture("x");
        if (x) {
            CMLPatternNode* inputs[1] = { x };
            CMLPatternNode* pat = cml_pattern_op(UOP_NEG, inputs, 1);
            if (pat) cml_rewrite_register(reg, pat, emit_fold_neg, 120, "fold_neg");
        }
    }

    /* ── Strength reduction rules (priority 95) ── */

    /* Rule 12: x * 2 -> x + x */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_MUL);
        if (pat) cml_rewrite_register(reg, pat, emit_mul_two, 95, "mul_two");
    }

    /* Rule 13: x * 0 -> FILL(0) */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_MUL);
        if (pat) cml_rewrite_register(reg, pat, emit_mul_zero, 95, "mul_zero");
    }

    /* Rule 14: x / const -> x * (1/const) */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_DIV);
        if (pat) cml_rewrite_register(reg, pat, emit_div_const, 95, "div_const");
    }

    /* ── Algebraic simplification rules (priority 80) ── */

    /* Rule 15: log(exp(x)) -> x */
    {
        CMLPatternNode* pat = make_unary_chain_pattern(UOP_LOG, UOP_EXP);
        if (pat) cml_rewrite_register(reg, pat, emit_log_exp, 80, "log_exp");
    }

    /* Rule 16: sqrt(x) * sqrt(x) -> x */
    {
        CMLPatternNode* pat = make_binop_pattern(UOP_MUL);
        if (pat) cml_rewrite_register(reg, pat, emit_sqrt_sq, 80, "sqrt_sq");
    }

    LOG_DEBUG("Built-in rewrite registry created with %d rules", reg->num_rules);
    return reg;
}

/* ================================================================
 * Dead Code Elimination (DCE)
 * ================================================================ */

/**
 * Mark-and-sweep DCE: starting from output nodes (the tail of the graph
 * and any nodes with live tensor references), mark all reachable nodes
 * via input_names, then remove unmarked nodes.
 */
int cml_rewrite_dce(CMLGraph_t ir) {
    if (!ir || !ir->head) return 0;

    /* Count nodes for allocation */
    int count = 0;
    struct IRNode* n = ir->head;
    while (n) { count++; n = n->next; }
    if (count == 0) return 0;

    /* Build node array for indexed access + visited flags */
    struct IRNode** nodes = malloc((size_t)count * sizeof(struct IRNode*));
    bool* marked = calloc((size_t)count, sizeof(bool));
    if (!nodes || !marked) { free(nodes); free(marked); return -1; }

    int idx = 0;
    n = ir->head;
    while (n) { nodes[idx++] = n; n = n->next; }

    /* Mark phase: start from tail + any node with a live tensor reference */
    /* Use a simple worklist approach */
    int* worklist = malloc((size_t)count * sizeof(int));
    int wl_head = 0, wl_tail = 0;
    if (!worklist) { free(nodes); free(marked); return -1; }

    for (int i = 0; i < count; i++) {
        /* Mark output nodes and nodes with live tensor references */
        bool is_live = (nodes[i] == ir->tail) ||
                       (nodes[i]->output && nodes[i]->output->ir_node == nodes[i]) ||
                       (nodes[i]->use_count > 0);
        if (is_live && !marked[i]) {
            marked[i] = true;
            worklist[wl_tail++] = i;
        }
    }

    /* BFS backwards through input dependencies */
    while (wl_head < wl_tail) {
        int ci = worklist[wl_head++];
        struct IRNode* cur = nodes[ci];

        for (int inp = 0; inp < cur->num_inputs; inp++) {
            if (!cur->input_names[inp]) continue;
            /* Find producer by output_name */
            for (int j = 0; j < count; j++) {
                if (!marked[j] && nodes[j]->output_name &&
                    strcmp(nodes[j]->output_name, cur->input_names[inp]) == 0) {
                    marked[j] = true;
                    worklist[wl_tail++] = j;
                    break;
                }
            }
        }
    }

    /* Sweep phase: remove unmarked nodes */
    int removed = 0;
    for (int i = 0; i < count; i++) {
        if (!marked[i]) {
            unlink_node(ir, nodes[i]);
            free_unlinked_node(nodes[i]);
            removed++;
        }
    }

    free(worklist);
    free(nodes);
    free(marked);

    if (removed > 0) {
        LOG_DEBUG("DCE: removed %d dead nodes", removed);
    }
    return removed;
}
