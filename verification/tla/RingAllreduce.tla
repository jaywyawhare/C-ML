------------------------- MODULE RingAllreduce -------------------------
(***************************************************************************)
(* Models the reduction phase of ring all-reduce over N ranks.             *)
(*                                                                          *)
(* Real subsystem: src/distributed/ring_allreduce.c. Multi-rank behavior   *)
(* is simulated in CI (test_multirank), never run across real nodes, so a   *)
(* step-count off-by-one in the ring is invisible to the test suite: the    *)
(* reduction still returns a number, just the wrong one.                     *)
(*                                                                          *)
(* A partial sum travels the ring accumulating each rank's contribution.    *)
(* A correct ring reduction visits every other rank, i.e. N-1 accumulation  *)
(* steps. CONSTANT Steps selects the loop bound.                            *)
(*                                                                          *)
(* Invariant Complete: once the reduction finishes, every rank's value has  *)
(* been folded in (so the accumulator equals the true total). TLC refutes   *)
(* it with Steps = N-2 (one contribution dropped) and verifies it with      *)
(* Steps = N-1.                                                             *)
(***************************************************************************)
EXTENDS Naturals
CONSTANTS N, Steps

ASSUME N \in Nat /\ N >= 2
ASSUME Steps \in Nat

Ranks == 0 .. (N - 1)

VARIABLES
    step,      \* accumulation steps taken so far
    included,  \* set of rank indices already folded into the partial sum
    acc        \* running sum (each rank i contributes value i for concreteness)

vars == <<step, included, acc>>

Total == (N * (N - 1)) \div 2   \* sum of rank values 0..N-1

Init ==
    /\ step = 0
    /\ included = {0}
    /\ acc = 0

\* Fold the next rank around the ring into the partial sum.
Accumulate ==
    /\ step < Steps
    /\ LET idx == step + 1 IN
         /\ idx \in Ranks
         /\ included' = included \cup {idx}
         /\ acc' = acc + idx
    /\ step' = step + 1

Done ==
    /\ step >= Steps
    /\ UNCHANGED vars

Next == Accumulate \/ Done

Spec == Init /\ [][Next]_vars

\* When the ring finishes, every rank must have been summed exactly once, so
\* the accumulator holds the true total.
Complete == (step = Steps) => (included = Ranks /\ acc = Total)
=============================================================================
