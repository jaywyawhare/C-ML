---------------------------- MODULE IRReset ----------------------------
(***************************************************************************)
(* Models C-ML's process-global IR graph versus cml_reset_* during a       *)
(* decode / training loop.                                                  *)
(*                                                                          *)
(* Real subsystem: src/core/computation_graph.c, src/cml.c.                 *)
(* Documented footgun: the IR graph accumulates process-globally; a reset   *)
(* frees every node, and a tensor created earlier in the step still points  *)
(* at a freed node. The observable crash is "Cannot compute gradients for   *)
(* NULL tensor" at backward time.                                           *)
(*                                                                          *)
(* The real fix is a keep-alive discipline (tensor_ensure_executed /        *)
(* tensor_detach_keep): inputs needed past a reset are realized/kept so a    *)
(* reset cannot free them. CONSTANT KeepInputs selects that discipline.      *)
(*                                                                          *)
(* Invariant SafeBackward: whenever the loop holds a forwarded tensor it    *)
(* still intends to use, that tensor is Live (not Dangling). TLC refutes it  *)
(* with KeepInputs = FALSE and verifies it with KeepInputs = TRUE.           *)
(***************************************************************************)
CONSTANT KeepInputs

VARIABLES
    phase,    \* "idle" -> "forwarded" -> "done" -> "idle" (continuous decode)
    xstate,   \* "empty" | "live" | "dangling": the input tensor's node state
    kept      \* TRUE once the input is realized/kept and survives a reset

vars == <<phase, xstate, kept>>

Init ==
    /\ phase = "idle"
    /\ xstate = "empty"
    /\ kept = FALSE

\* Forward builds the graph: the input becomes Live. Under the keep-alive
\* discipline it is realized in the same step so a later reset cannot free it.
Forward ==
    /\ phase = "idle"
    /\ phase' = "forwarded"
    /\ xstate' = "live"
    /\ kept' = KeepInputs

\* A process-global reset can fire at any time (another op triggers it mid
\* decode). It frees every un-kept node; a Live-but-unkept input dangles.
Reset ==
    /\ xstate' = IF (xstate = "live" /\ ~kept) THEN "dangling" ELSE xstate
    /\ UNCHANGED <<phase, kept>>

\* Backward consumes the input, then the loop comes back around.
Backward ==
    /\ phase = "forwarded"
    /\ phase' = "done"
    /\ UNCHANGED <<xstate, kept>>

Restart ==
    /\ phase = "done"
    /\ phase' = "idle"
    /\ xstate' = "empty"
    /\ kept' = FALSE

Next == Forward \/ Reset \/ Backward \/ Restart

Spec == Init /\ [][Next]_vars

\* The hazard: holding a forwarded input that a reset has turned into a
\* dangling (NULL) node. Backward on it is the documented crash.
SafeBackward == (phase = "forwarded") => (xstate # "dangling")
=============================================================================
