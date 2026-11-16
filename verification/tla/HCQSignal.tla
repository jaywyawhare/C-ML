--------------------------- MODULE HCQSignal ---------------------------
(***************************************************************************)
(* Models the HCQ producer/consumer signal dependency that every backend   *)
(* shares.                                                                   *)
(*                                                                          *)
(* Real subsystem: src/ops/ir/hcq.c, hcq_backend.c. HCQ is marked Stable    *)
(* but is only ever exercised on the CPU reference backend, i.e. one        *)
(* interleaving. On real hardware the host enqueue and device execution     *)
(* run concurrently.                                                         *)
(*                                                                          *)
(* A command writes a buffer and posts a completion signal. A dependent     *)
(* command must wait on that signal before reading the buffer. CONSTANT      *)
(* WaitForSignal selects whether the consumer honors the dependency.         *)
(*                                                                          *)
(* Invariant NoEarlyRead: the consumer never reads the buffer before the    *)
(* producer has written it. TLC refutes it with WaitForSignal = FALSE (a    *)
(* read races ahead of the write) and verifies it with WaitForSignal = TRUE.*)
(***************************************************************************)
CONSTANT WaitForSignal

VARIABLES
    written,    \* producer has written the buffer
    posted,     \* producer has posted the completion signal (only after write)
    readDone,   \* consumer has performed its read
    readSawData \* whether the buffer was already written at read time

vars == <<written, posted, readDone, readSawData>>

Init ==
    /\ written = FALSE
    /\ posted = FALSE
    /\ readDone = FALSE
    /\ readSawData = FALSE

\* Producer step 1: write the output buffer.
Write ==
    /\ ~written
    /\ written' = TRUE
    /\ UNCHANGED <<posted, readDone, readSawData>>

\* Producer step 2: post the signal. Hardware guarantees this is ordered
\* after the write that precedes it in the queue.
Post ==
    /\ written
    /\ ~posted
    /\ posted' = TRUE
    /\ UNCHANGED <<written, readDone, readSawData>>

\* Consumer: read the buffer. With the dependency honored it may only fire
\* once the producer's signal is posted; without it, it may fire any time.
Read ==
    /\ ~readDone
    /\ (WaitForSignal => posted)
    /\ readDone' = TRUE
    /\ readSawData' = written
    /\ UNCHANGED <<written, posted>>

\* Terminal stutter once the read is done, so a finished run is not reported
\* as a deadlock (termination here is success, not a stuck protocol).
Terminate ==
    /\ readDone
    /\ UNCHANGED vars

Next == Write \/ Post \/ Read \/ Terminate

Spec == Init /\ [][Next]_vars

\* Once the consumer has read, the buffer must have been written first.
NoEarlyRead == readDone => readSawData
=============================================================================
