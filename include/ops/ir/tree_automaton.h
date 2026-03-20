#pragma once
#include "ops/ir/pattern_matcher.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLAutomatonState CMLAutomatonState;
typedef struct CMLAutomaton CMLAutomaton;

CMLAutomaton* cml_automaton_compile(CMLRewriteRegistry* registry);
void cml_automaton_free(CMLAutomaton* automaton);

int cml_automaton_rewrite(CMLAutomaton* automaton, struct CMLGraph* graph);

int cml_automaton_num_states(const CMLAutomaton* automaton);
int cml_automaton_num_transitions(const CMLAutomaton* automaton);

#ifdef __cplusplus
}
#endif
