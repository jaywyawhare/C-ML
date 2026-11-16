# Formal verification

TLA+ models of the C-ML protocols whose bugs cost the most and whose
interleavings the C test suite cannot enumerate. Each model is checked in two
configurations: a `*_correct.cfg` that must hold, and a `*_broken.cfg` with one
planted defect that TLC must refute. A spec that cannot catch its own planted
bug proves nothing, so `run.sh` asserts both halves.

## Run

```bash
TLA2TOOLS=/path/to/tla2tools.jar ./verification/run.sh
```

`run.sh` also finds the jar at `$HOME/.cache/tla/tla2tools.jar`. The nightly
`formal` job in `.github/workflows/fuzz.yml` fetches a pinned TLC and runs this.

## Models

| Spec | Subsystem | Invariant | Planted bug (broken.cfg) |
|------|-----------|-----------|--------------------------|
| `IRReset` | `src/core/computation_graph.c`, `src/cml.c` | A forwarded input tensor is never left dangling when a process-global `cml_reset_*` fires mid-decode | Keep-alive discipline off: the reset frees the un-realized input, reproducing the "Cannot compute gradients for NULL tensor" crash |
| `HCQSignal` | `src/ops/ir/hcq.c`, `hcq_backend.c` | A dependent command never reads a buffer before the producing command has written it | Consumer skips the signal wait, so a read races ahead of the write |
| `RingAllreduce` | `src/distributed/ring_allreduce.c` | When the ring reduction finishes, every rank's contribution has been folded in exactly once | Off-by-one step bound (`N-2` instead of `N-1`) silently drops one rank |

## Bounds

These are finite-state abstractions of the protocols, not of the C source.
They catch ordering, lifecycle and step-count defects in the design; they do
not catch a C typo or a buffer overrun, which the sanitizer and fuzz legs cover.

- `IRReset`: one input tensor, one reset interleaving per decode iteration,
  continuous loop. State space is small and fully enumerated.
- `HCQSignal`: one producer, one consumer, single buffer and signal.
- `RingAllreduce`: `N = 4` ranks (`N-1 = 3` steps). The property generalizes to
  any `N`; `N = 4` is the smallest bound that distinguishes the off-by-one.

When a modeled file changes, re-run `run.sh` and update the spec if the protocol
changed (see the `maintain-formal-verification-skill`).
