#include "ops/ir/tree_automaton.h"
#include "ops/ir/internal.h"
#include "ops/ir/pattern_matcher.h"
#include "ops/uops.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>

#define AUTOMATON_INITIAL_TABLE_CAP 256
#define AUTOMATON_MAX_ARITY CML_PATTERN_MAX_INPUTS
#define AUTOMATON_DEAD_STATE 0
#define AUTOMATON_WILDCARD_STATE 1
#define AUTOMATON_INITIAL_STATES_CAP 64

#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL

struct CMLAutomatonState {
    int id;
    int* matched_rules;
    int num_matched;
};

typedef struct {
    uint64_t key;
    int result_state;
    bool occupied;
} CMLTransition;

struct CMLAutomaton {
    CMLAutomatonState* states;
    int num_states;
    int states_capacity;
    CMLTransition* transitions;
    int num_transitions;
    int table_capacity;
    CMLRewriteRegistry* registry;
    int wildcard_state;
};

static uint64_t fnv1a_init(void) { return FNV_OFFSET_BASIS; }

static uint64_t fnv1a_int(uint64_t hash, int value) {
    const uint8_t* bytes = (const uint8_t*)&value;
    for (size_t i = 0; i < sizeof(int); i++) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

static uint64_t transition_key(int op_type, const int* child_states, int arity) {
    uint64_t h = fnv1a_init();
    h = fnv1a_int(h, op_type);
    h = fnv1a_int(h, arity);
    for (int i = 0; i < arity; i++)
        h = fnv1a_int(h, child_states[i]);
    return h;
}

static int transition_lookup(const CMLAutomaton* aut, uint64_t key) {
    if (aut->table_capacity == 0) return AUTOMATON_DEAD_STATE;
    uint64_t idx = key % (uint64_t)aut->table_capacity;
    for (int probe = 0; probe < aut->table_capacity; probe++) {
        uint64_t slot = (idx + (uint64_t)probe) % (uint64_t)aut->table_capacity;
        if (!aut->transitions[slot].occupied)
            return AUTOMATON_DEAD_STATE;
        if (aut->transitions[slot].key == key)
            return aut->transitions[slot].result_state;
    }
    return AUTOMATON_DEAD_STATE;
}

static void transition_grow(CMLAutomaton* aut);

static void transition_insert(CMLAutomaton* aut, uint64_t key, int result_state) {
    if (aut->num_transitions * 2 >= aut->table_capacity)
        transition_grow(aut);

    uint64_t idx = key % (uint64_t)aut->table_capacity;
    for (int probe = 0; probe < aut->table_capacity; probe++) {
        uint64_t slot = (idx + (uint64_t)probe) % (uint64_t)aut->table_capacity;
        if (!aut->transitions[slot].occupied) {
            aut->transitions[slot].key = key;
            aut->transitions[slot].result_state = result_state;
            aut->transitions[slot].occupied = true;
            aut->num_transitions++;
            return;
        }
        if (aut->transitions[slot].key == key) {
            aut->transitions[slot].result_state = result_state;
            return;
        }
    }
}

static void transition_grow(CMLAutomaton* aut) {
    int old_cap = aut->table_capacity;
    CMLTransition* old = aut->transitions;

    int new_cap = old_cap < 16 ? 16 : old_cap * 2;
    aut->transitions = calloc((size_t)new_cap, sizeof(CMLTransition));
    aut->table_capacity = new_cap;
    aut->num_transitions = 0;

    for (int i = 0; i < old_cap; i++) {
        if (old[i].occupied)
            transition_insert(aut, old[i].key, old[i].result_state);
    }
    free(old);
}

static int automaton_add_state(CMLAutomaton* aut) {
    if (aut->num_states >= aut->states_capacity) {
        int new_cap = aut->states_capacity * 2;
        if (new_cap < 16) new_cap = 16;
        CMLAutomatonState* ns = realloc(aut->states, (size_t)new_cap * sizeof(CMLAutomatonState));
        if (!ns) return -1;
        aut->states = ns;
        aut->states_capacity = new_cap;
    }
    int id = aut->num_states;
    CMLAutomatonState* s = &aut->states[id];
    s->id = id;
    s->matched_rules = NULL;
    s->num_matched = 0;
    aut->num_states++;
    return id;
}

static void state_add_rule(CMLAutomaton* aut, int state_id, int rule_idx) {
    if (state_id < 0 || state_id >= aut->num_states) return;
    CMLAutomatonState* s = &aut->states[state_id];

    for (int i = 0; i < s->num_matched; i++) {
        if (s->matched_rules[i] == rule_idx)
            return;
    }

    s->matched_rules = realloc(s->matched_rules,
                                (size_t)(s->num_matched + 1) * sizeof(int));
    s->matched_rules[s->num_matched] = rule_idx;
    s->num_matched++;
}

/*
 * Build a state for a pattern subtree. Returns the state id that the
 * subtree transitions into. For CML_PAT_OP nodes this creates transitions
 * from child states to a new state; CML_PAT_ANY and CML_PAT_CAPTURE both
 * map to the wildcard state.
 */
static int compile_pattern_node(CMLAutomaton* aut, const CMLPatternNode* pat,
                                int rule_idx, bool is_root) {
    if (!pat) return AUTOMATON_DEAD_STATE;

    switch (pat->kind) {
    case CML_PAT_ANY:
    case CML_PAT_CAPTURE:
        return aut->wildcard_state;

    case CML_PAT_OP: {
        int child_states[AUTOMATON_MAX_ARITY];
        for (int i = 0; i < pat->num_inputs; i++) {
            child_states[i] = compile_pattern_node(aut, pat->inputs[i], rule_idx, false);
        }

        uint64_t key = transition_key((int)pat->op_type, child_states, pat->num_inputs);
        int existing = transition_lookup(aut, key);
        if (existing != AUTOMATON_DEAD_STATE) {
            if (is_root)
                state_add_rule(aut, existing, rule_idx);
            return existing;
        }

        int new_state = automaton_add_state(aut);
        if (new_state < 0) return AUTOMATON_DEAD_STATE;
        transition_insert(aut, key, new_state);

        if (is_root)
            state_add_rule(aut, new_state, rule_idx);

        return new_state;
    }

    default:
        return AUTOMATON_DEAD_STATE;
    }
}

/*
 * For patterns containing wildcards, we need to expand transitions so that
 * the wildcard state matches any actual child state. This generates
 * additional transitions for every (op, child-state-combo) that could arise
 * when wildcard positions are filled by concrete states.
 *
 * We iterate over all patterns and for each OP-node whose children include
 * a wildcard, we insert a transition for every currently-known concrete
 * state at that child position. This is repeated until no new transitions
 * are added (fixpoint).
 */
typedef struct {
    int op_type;
    int arity;
    int child_pattern_states[AUTOMATON_MAX_ARITY];
    int target_state;
    int rule_idx;
    bool is_root;
} PendingWildcard;

static void expand_wildcard_transitions(CMLAutomaton* aut,
                                        PendingWildcard* pending, int num_pending) {
    if (num_pending == 0) return;

    bool changed = true;
    while (changed) {
        changed = false;
        for (int p = 0; p < num_pending; p++) {
            PendingWildcard* pw = &pending[p];
            bool has_wildcard = false;
            for (int c = 0; c < pw->arity; c++) {
                if (pw->child_pattern_states[c] == aut->wildcard_state)
                    has_wildcard = true;
            }
            if (!has_wildcard) continue;

            /* Enumerate all concrete states to substitute for wildcards.
             * For simplicity we try each concrete state in each wildcard
             * position independently (not full cartesian product -- that
             * would be exponential). For the patterns in this codebase
             * (arity <= 2, usually only leaves are wildcards) this is
             * sufficient. */
            for (int c = 0; c < pw->arity; c++) {
                if (pw->child_pattern_states[c] != aut->wildcard_state)
                    continue;
                for (int sid = 2; sid < aut->num_states; sid++) {
                    int trial[AUTOMATON_MAX_ARITY];
                    memcpy(trial, pw->child_pattern_states, sizeof(trial));
                    trial[c] = sid;

                    uint64_t key = transition_key(pw->op_type, trial, pw->arity);
                    int existing = transition_lookup(aut, key);
                    if (existing == AUTOMATON_DEAD_STATE) {
                        transition_insert(aut, key, pw->target_state);
                        changed = true;
                    } else if (existing != pw->target_state && pw->is_root) {
                        state_add_rule(aut, existing, pw->rule_idx);
                    }
                }
            }
        }
    }
}

CMLAutomaton* cml_automaton_compile(CMLRewriteRegistry* registry) {
    if (!registry || registry->num_rules == 0) return NULL;

    CMLAutomaton* aut = calloc(1, sizeof(CMLAutomaton));
    if (!aut) return NULL;

    aut->registry = registry;
    aut->states = NULL;
    aut->num_states = 0;
    aut->states_capacity = 0;
    aut->transitions = NULL;
    aut->num_transitions = 0;
    aut->table_capacity = 0;

    /* State 0 = dead, state 1 = wildcard */
    automaton_add_state(aut); /* dead */
    automaton_add_state(aut); /* wildcard */
    aut->wildcard_state = AUTOMATON_WILDCARD_STATE;

    /* Allocate pending wildcard records for expansion pass */
    int pending_cap = registry->num_rules * AUTOMATON_MAX_ARITY;
    PendingWildcard* pending = calloc((size_t)pending_cap, sizeof(PendingWildcard));
    int num_pending = 0;

    for (int r = 0; r < registry->num_rules; r++) {
        CMLRewriteRule* rule = &registry->rules[r];
        if (!rule->pattern) continue;

        int root_state = compile_pattern_node(aut, rule->pattern, r, true);
        (void)root_state;

        /* Record wildcard positions for root-level OP patterns */
        if (rule->pattern->kind == CML_PAT_OP && num_pending < pending_cap) {
            PendingWildcard* pw = &pending[num_pending];
            pw->op_type = (int)rule->pattern->op_type;
            pw->arity = rule->pattern->num_inputs;
            pw->target_state = root_state;
            pw->rule_idx = r;
            pw->is_root = true;
            for (int c = 0; c < rule->pattern->num_inputs; c++) {
                pw->child_pattern_states[c] =
                    compile_pattern_node(aut, rule->pattern->inputs[c], r, false);
            }
            num_pending++;
        }
    }

    expand_wildcard_transitions(aut, pending, num_pending);
    free(pending);

    return aut;
}

void cml_automaton_free(CMLAutomaton* automaton) {
    if (!automaton) return;
    for (int i = 0; i < automaton->num_states; i++)
        free(automaton->states[i].matched_rules);
    free(automaton->states);
    free(automaton->transitions);
    free(automaton);
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

/*
 * Compute the automaton state for a single IR node given its children's
 * states. Tries the exact (op, child_states) key first. If that misses
 * and some children have the wildcard state, falls back to wildcard
 * expansion by trying the wildcard state in each child position.
 */
static int compute_node_state(const CMLAutomaton* aut, struct IRNode* node,
                              const int* child_states, int arity) {
    uint64_t key = transition_key((int)node->type, child_states, arity);
    int state = transition_lookup(aut, key);
    if (state != AUTOMATON_DEAD_STATE)
        return state;

    /* Try with wildcard in each position as fallback */
    int trial[AUTOMATON_MAX_ARITY];
    memcpy(trial, child_states, (size_t)arity * sizeof(int));

    /* Single-wildcard fallback per position */
    for (int i = 0; i < arity; i++) {
        int saved = trial[i];
        trial[i] = AUTOMATON_WILDCARD_STATE;
        key = transition_key((int)node->type, trial, arity);
        state = transition_lookup(aut, key);
        if (state != AUTOMATON_DEAD_STATE)
            return state;
        trial[i] = saved;
    }

    /* All-wildcard fallback */
    for (int i = 0; i < arity; i++)
        trial[i] = AUTOMATON_WILDCARD_STATE;
    key = transition_key((int)node->type, trial, arity);
    return transition_lookup(aut, key);
}

static atomic_int g_rewrite_counter = 0;

static char* rewrite_unique_name(void) {
    int id = atomic_fetch_add(&g_rewrite_counter, 1);
    char* name = malloc(32);
    if (name)
        snprintf(name, 32, "_rw%d", id);
    return name;
}

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

static void insert_node_before(CMLGraph_t ir, struct IRNode* new_node,
                               struct IRNode* before) {
    if (!ir || !new_node) return;
    new_node->next = NULL;

    if (!before || !ir->head) {
        if (ir->tail) {
            ir->tail->next = new_node;
        } else {
            ir->head = new_node;
        }
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
        ir->node_count++;
    } else {
        ir->tail->next = new_node;
        ir->tail = new_node;
        ir->node_count++;
    }
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

static void free_unlinked_node(struct IRNode* node) {
    if (!node) return;

    if (node->input_names) {
        for (int i = 0; i < node->num_inputs; i++)
            free(node->input_names[i]);
        free(node->input_names);
    }
    free(node->output_name);
    free(node->users);

    if (node->output) {
        node->output->ir_node = NULL;
        node->output->ir_context = NULL;
    }

    free(node);
}

static bool match_node_recursive(CMLGraph_t ir, const CMLPatternNode* pattern,
                                 struct IRNode* node, CMLMatchResult* result) {
    if (!pattern || !node) return false;

    switch (pattern->kind) {
    case CML_PAT_ANY:
        return true;
    case CML_PAT_CAPTURE: {
        if (!result) return false;
        if (result->num_captures >= CML_PATTERN_MAX_CAPTURES) return false;
        for (int i = 0; i < result->num_captures; i++) {
            if (strcmp(result->captures[i].name, pattern->capture_name) == 0)
                return (result->captures[i].node == node);
        }
        CMLCaptureEntry* e = &result->captures[result->num_captures];
        strncpy(e->name, pattern->capture_name, sizeof(e->name) - 1);
        e->name[sizeof(e->name) - 1] = '\0';
        e->node = node;
        result->num_captures++;
        return true;
    }
    case CML_PAT_OP: {
        if (node->type != pattern->op_type) return false;
        if (pattern->num_inputs != node->num_inputs) return false;
        for (int i = 0; i < pattern->num_inputs; i++) {
            if (!pattern->inputs[i]) return false;
            struct IRNode* producer = find_node_by_output(ir, node->input_names[i]);
            if (!producer) return false;
            if (!match_node_recursive(ir, pattern->inputs[i], producer, result))
                return false;
        }
        return true;
    }
    default:
        return false;
    }
}

int cml_automaton_rewrite(CMLAutomaton* automaton, struct CMLGraph* graph) {
    if (!automaton || !graph) return -1;
    if (!automaton->registry || automaton->registry->num_rules == 0) return 0;

    int max_iter = CML_REWRITE_DEFAULT_MAX_ITER;
    int total_rewrites = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Phase 1: topological order (children before parents) */
        int cap = graph->node_count > 0 ? graph->node_count : 16;
        struct IRNode** topo = malloc((size_t)cap * sizeof(struct IRNode*));
        int* node_states = calloc((size_t)cap, sizeof(int));
        if (!topo || !node_states) { free(topo); free(node_states); return -1; }

        int topo_count = 0;
        struct IRNode* n = graph->head;
        while (n) {
            if (topo_count >= cap) {
                cap *= 2;
                topo = realloc(topo, (size_t)cap * sizeof(struct IRNode*));
                node_states = realloc(node_states, (size_t)cap * sizeof(int));
                if (!topo || !node_states) { free(topo); free(node_states); return -1; }
            }
            topo[topo_count] = n;
            node_states[topo_count] = AUTOMATON_DEAD_STATE;
            topo_count++;
            n = n->next;
        }

        /* Phase 2: bottom-up state assignment.
         * The graph is stored in topological order (producers before
         * consumers) due to the way the IR builder appends nodes, so
         * iterating forward assigns children before parents. */
        for (int i = 0; i < topo_count; i++) {
            struct IRNode* node = topo[i];
            int child_states[AUTOMATON_MAX_ARITY] = {0};
            int arity = node->num_inputs < AUTOMATON_MAX_ARITY
                            ? node->num_inputs : AUTOMATON_MAX_ARITY;

            for (int c = 0; c < arity; c++) {
                if (!node->input_names[c]) continue;
                struct IRNode* producer = find_node_by_output(graph, node->input_names[c]);
                if (!producer) continue;

                /* Find producer's index to get its state */
                for (int j = 0; j < topo_count; j++) {
                    if (topo[j] == producer) {
                        child_states[c] = node_states[j];
                        break;
                    }
                }
            }

            node_states[i] = compute_node_state(automaton, node, child_states, arity);
        }

        /* Phase 3: apply rewrites for accepting states */
        int rewrites_this_pass = 0;

        for (int i = 0; i < topo_count; i++) {
            int state = node_states[i];
            if (state <= AUTOMATON_WILDCARD_STATE) continue;
            if (state >= automaton->num_states) continue;

            CMLAutomatonState* as = &automaton->states[state];
            if (as->num_matched == 0) continue;

            struct IRNode* node = topo[i];
            bool replaced = false;

            for (int m = 0; m < as->num_matched && !replaced; m++) {
                int rule_idx = as->matched_rules[m];
                if (rule_idx < 0 || rule_idx >= automaton->registry->num_rules)
                    continue;

                CMLRewriteRule* rule = &automaton->registry->rules[rule_idx];

                CMLMatchResult result;
                memset(&result, 0, sizeof(result));
                result.matched_root = node;

                if (!match_node_recursive(graph, rule->pattern, node, &result))
                    continue;

                struct IRNode* replacement = rule->emit(graph, &result);
                if (!replacement || replacement == node) continue;

                if (!replacement->output_name)
                    replacement->output_name = rewrite_unique_name();

                bool already_in_graph = false;
                {
                    struct IRNode* scan = graph->head;
                    while (scan) {
                        if (scan == replacement) { already_in_graph = true; break; }
                        scan = scan->next;
                    }
                }
                if (!already_in_graph)
                    insert_node_before(graph, replacement, node);

                if (node->output_name && replacement->output_name)
                    replace_output_references(graph, node->output_name,
                                              replacement->output_name);

                if (node->output && replacement->output) {
                    node->output->ir_node = replacement;
                    node->output->ir_context = graph;
                }

                unlink_node(graph, node);
                free_unlinked_node(node);
                topo[i] = NULL;

                rewrites_this_pass++;
                replaced = true;
            }
        }

        free(topo);
        free(node_states);

        total_rewrites += rewrites_this_pass;
        if (rewrites_this_pass == 0)
            break;
    }

    return total_rewrites;
}

int cml_automaton_num_states(const CMLAutomaton* automaton) {
    return automaton ? automaton->num_states : 0;
}

int cml_automaton_num_transitions(const CMLAutomaton* automaton) {
    return automaton ? automaton->num_transitions : 0;
}
