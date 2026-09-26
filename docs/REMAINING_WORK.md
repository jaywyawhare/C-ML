# Remaining Work Items

## 1. GPU Backward Pass with Per-Node CPU Fallback ✅ DONE

`cml_vulkan_execute_graph()` falls back to `cpu_execute_node()` per node instead
of failing the whole graph. Implemented and tested.

---

## 2. GPU Codegen Elementwise Op Coverage ✅ DONE

The unary/binary op gaps in the GPU codegen are closed in **both** backends,
with the two op tables re-synced (PTX previously lacked TAN):

- `src/ops/ir/gpu/gpu_codegen.c` (LLVM IR): added unary GELU, QUICK_GELU,
  LEAKY_RELU, HARD_SIGMOID, HARD_TANH, RELU6, SQUARE, RSQRT, EXP2, LOG2, FLOOR,
  CEIL, SIGN; binary MINIMUM, CMPGT, CMPGE, CMPLE, CMPEQ, CMPNE.
- `src/ops/ir/gpu/ptx_codegen.c` (PTX text): same set plus TAN, using
  `float_to_ptx_hex` for the irrational activation constants so the emitted
  bit patterns match the CPU path.

Semantics (NaN handling, LEAKY_RELU/ELU fixed default slopes, tanh-approx GELU)
follow `src/ops/ir/execution_typed.c`. Coverage added in
`tests/test_ptx_codegen.c` (`test_new_unary_ops`, `test_new_binary_ops`). The
numeric `tests/test_gpu_codegen.c` cases remain hardware-gated (skip without a
CUDA/ROCm device); PTX text emission is the portable validation.

---

## 3. Other closed gaps (this pass)

- **ONNX importer** (`src/core/onnx_ops.c`): CumSum `exclusive`/`reverse`
  (composed from the inclusive scan) and Resize `align_corners` /
  `pytorch_half_pixel` coordinate modes. Tests in `test_onnx_import_ops.c`.
- **AOT** (`src/ops/ir/aot.c`): general (non-trivial) EXPAND broadcast and
  multi-dim partial reductions. Tests in `test_aot_roundtrip.c`.
- **DDP** (`src/distributed/data_parallel.c`): `find_unused_parameters` now
  reserves a zero-filled bucket slot for gradient-less params so every rank's
  bucket layout matches (the all-reduce would otherwise sum mismatched slots).
- **`FUSE_OPTIM`** (`src/optim.c`): the SGD step now emits every parameter's
  `uop_sgd_step` into the graph and realizes them in a single co-scheduled pass
  (idempotent executor => no double-applied momentum), instead of realizing each
  update on its own. `test_fuse_optim_matches_default` in `test_optim.c` trains
  identical models with and without the flag and requires the weights to match;
  the convergence suite passes under `FUSE_OPTIM=1`. Other optimizers (Adam,
  etc.) still realize per parameter.
- **Async disk I/O** (`src/backend/disk_backend.c`): real io_uring behind
  `CML_HAS_IO_URING` (CMake-detected liburing), synchronous fallback otherwise.
  Test in `test_disk_backend.c`.
- **OpenCL batched matmul** (`src/ops/ir/gpu/opencl_ir_backend.c`): GPU path via
  per-batch sub-buffer views over the reused 2D GEMM; falls back to CPU when a
  slice offset isn't device-aligned. Test in `test_opencl_ir.c`.

---

## 4. Double-backward / `create_graph` ✅ ALREADY DONE (graph mode)

Second-order gradients work under the graph autodiff engine: `tensor_backward`
threads `create_graph` into `cml_ir_grad` and skips backward-subgraph fusion so a
second pass can re-differentiate (`autodiff.c:1067`). Verified end-to-end in
`tests/test_double_backward.c` (exact `d²/dx²`, finite-diff for quadratic and
matmul, composite-VJP, and the no-create_graph-inert case). The *eager* engine
cannot support it — gradients are plain data with nothing to differentiate — and
it says so loudly (`autograd.c:319`); that is a fundamental engine property, not a
gap.

## 5. Closed in this pass

- **`gradient_as_bucket_view`** (`src/distributed/data_parallel.c`) — honored.
  Each gradient's storage is repointed at its slot in the flat bucket
  (`ddp_alias_grad_to_slot`), so the pack/unpack memcpys around the all-reduce
  and the separate per-gradient allocations both go away. The ownership hazard
  the option carries is handled explicitly rather than avoided:
  - `tensor_realize()` runs first, which both materialises a lazy gradient and
    detaches it from the IR graph — a still-attached gradient could be
    re-executed into a fresh buffer later and silently drop the alias.
  - The aliased tensor gets `owns_data = false`, so `tensor_free` leaves the
    bucket's memory alone (no double free).
  - The option implies the reserved bucket layout that `find_unused_parameters`
    requests: an alias cannot survive offsets that shift between steps.
  - `cml_ddp_free()` copies every aliased gradient back into its own allocation
    before releasing the buckets, so gradients stay valid after the wrapper dies
    (no use-after-free). It only un-aliases pointers that really fall inside its
    own buckets, so a gradient replaced since the last sync is left alone.
  - A gradient that cannot be aliased (device memory, non-float32, size
    mismatch) falls back to copying, tracked per parameter in `grad_is_view`.

  Contrary to the previous note here, this **is** validatable without MPI: the
  alias is a storage layout, not a collective, so `world_size == 1` exercises all
  of it. `tests/test_ddp_bucket_view.c` checks that each gradient's data pointer
  *is* its bucket slot, that values are bit-identical to the copying path, that
  re-syncing is stable, and that gradients survive `cml_ddp_free`. Run under
  `-DENABLE_SANITIZERS=ON` for the double-free half.

- **Pipeline `interleaved` schedule** (`src/distributed/pipeline_parallel.c`) —
  honored. `cml_pipeline_build_schedule()` emits the execution order as data and
  both `cml_pipeline_forward` and `cml_pipeline_backward` now run their units in
  it. GPipe is every forward then every backward; `interleaved` is 1F1B, built by
  replaying each stage's warmup/steady/drain program and emitting whichever
  stage's next unit has its dependencies met (backwards preferred, deepest stage
  first). Under 1F1B a micro-batch's cached activations are released the moment
  its backward reaches stage 0, which bounds live activations by the pipeline
  depth instead of the micro-batch count — so an interleaved backward consumes
  the forward's cache and cannot be re-run against it.

  Weight gradients accumulate over micro-batches, so the order cannot change the
  sum: `tests/test_pipeline_schedule.c` checks both schedules are valid
  topological orders across a range of (stages, micro-batches), that 1F1B's peak
  live-activation count is strictly lower and bounded by the depth, and that
  training through the two lands on **bitwise identical** gradients.

## 6. Closed in the follow-up pass

Five gaps that were **not** tracked in this document, found by auditing the
"not implemented"/stub markers rather than the list above.

- **ONNX importer read every integer operand as float32**
  (`src/core/onnx_ops.c`). ONNX passes shapes, axes, indices, pads and slice
  bounds as INT64 tensors. Reading those bytes through a `float*` is not a
  rounding problem: the low four bytes of a small little-endian int64 are a
  float32 denormal, so `2` came back as `2.8e-45` — zero. `Reshape` therefore
  computed a target shape of all-zeros, which its "0 means keep the input's dim"
  rule quietly turned into the *input* shape, and the op failed on a numel
  mismatch. `Expand`, `Tile`, `Slice`, `Squeeze`, `Unsqueeze`, `Pad`, `CumSum`,
  `Split`, `ScatterND`, `Resize` and the reduce family all read operands the same
  way. Added `operand_int`/`operand_int_list`, which dispatch on the tensor's own
  dtype, and converted every integer-operand site. Any model from a real exporter
  (PyTorch included) hits this path, so it was not a corner case.

- **Two initializer paths disagreed about their own dtype**
  (`src/core/onnx.c`). `int64_data` decoded into a *float* buffer while building
  the tensor with the declared dtype (INT64), so the tensor lied about its
  representation; `raw_data` stored genuine int64 bytes. Consumers reading by
  dtype therefore saw garbage from one path and correct values from the other.
  `int64_data` now decodes to real `int64_t`.

- **`emit_shape_initializer` wrote the wrong `dims`** (`src/core/onnx_export.c`).
  A shape operand is a 1-D int64 tensor of `ndim` values, but the exporter passed
  the shape values *as* `dims` — declaring a tensor of `prod(shape)` int64s while
  supplying only `ndim` of them. Every exported `Reshape`/`Expand` was unloadable.
  Now emits `dims = [ndim]`. With the two fixes above, `Reshape` round-trips
  export → import → run for the first time (`test_onnx_export.c`).

- **Partial-range `Flatten` export** (`src/core/onnx_export.c`) — ONNX `Flatten`
  always runs to the last dim, so a partial range has no single-op equivalent and
  the exporter failed outright. It now lowers to an explicit `Reshape` using the
  node's known output shape. The full-tail case still emits the cheaper plain
  `Flatten`. The old partial/full test was also simply wrong: it compared
  `end_dim` (an *input* dim index) against the *output* rank, which only coincide
  when exactly two dims collapse, so valid flatten-to-end nodes were
  misclassified. Rank now comes from `input_ndims`, the snapshot that survives
  realization (the `inputs` Tensor* array can be cleared).

- **Thunder executor advertised twelve ops it could not dispatch**
  (`src/backend/thunder_executor.c`). `op_table` rows resolved by name and then
  fell through the switch to a confusing "dispatch not implemented". Eight are now
  implemented — `SIGN FLOOR CEIL ROUND ERF POW`, plus `MAX_REDUCE` (NULL
  `ReduceParams` reduces every axis, as `SUM`/`MEAN` already did here) and `WHERE`
  (its operands come from a params struct, not positionally). The remaining four —
  `RESHAPE PERMUTE CONV2D GATHER` — need attributes (`target shape`,
  permutation, stride/padding, gather dim) that `CMLThunderOp` has no field for
  and that are not derivable from the inputs, so they were **removed from the
  table**: a row is a claim the op is dispatchable, and the clear "Unsupported op"
  path now fires instead. This follows the convention the file already used for
  `torch.gelu`/`torch.leaky_relu`. First tests for the executor
  (`test_thunder_executor.c`), including one that walks every advertised name.

- **Gradient checkpointing reclaimed no memory**
  (`src/autograd/checkpointing.c`). `autograd_checkpoint` saved the IR linkage and
  detached the node but never released `tensor->data` — so the feature paid the
  recompute cost and saved nothing, which is the one thing it exists to do. It now
  frees the activation (honouring `storage` / buffer-cache ownership, leaving
  backend-owned device buffers alone), and `autograd_recompute` allocates a fresh
  buffer before restoring, since there is no longer one to copy into. Saved input
  pointers are also pinned now and released in cleanup; they were raw `Tensor*`
  dereferenced long after the forward pass. First tests for the file
  (`test_checkpointing.c`).

- **Python DDP and pipeline never called the C engine**
  (`python/cml/distributed.py`). `DistributedDataParallel.__init__` stored
  `bucket_size_mb` and never called `cml_ddp_create` — no rank-0 broadcast, no
  bucketing, and neither `find_unused_parameters` nor the `gradient_as_bucket_view`
  from §5 was reachable; `sync_gradients` hand-rolled an unbucketed per-parameter
  all-reduce and then discarded the in-place result. `PipelineParallel.__call__`
  just chained the modules, so `num_micro_batches` was inert and `interleaved`
  unreachable. Both now wrap the real handles, with `build_pipeline_schedule()`
  exposing the schedule as data. The cffi `cdef` gained the DDP/pipeline
  declarations. Also removed the `_setup_distributed_bindings` ctypes scaffolding:
  `_get_lib()` returns the **cffi** lib, so every `.argtypes` assignment there
  raised `AttributeError` into a bare `except` — the whole block was a no-op.
  24 tests in `python/tests/test_distributed_py.py`.

- **`multi_schedule_run` reported success for work it never did**
  (`src/ops/ir/schedule_multi.c`). `CMLSchedule` has no executor, so device-compute
  steps cannot run, yet the function returned 0. It now returns -2 when the
  schedule contains compute steps, with the contract documented on the header.
  The partitioning and cost figures (`multi_schedule_build`,
  `xfer_bytes`, `total_kernels`) are computed for real and now have tests
  (`test_schedule_multi.c`); the API previously had none and no callers.

Note on test counts: a hardware-gated skip is reported as a pass, so the suite
total overstates coverage on a machine without a GPU — `test_gpu_codegen` prints
8/8 with every case skipped. `test_opencl_ir` does run for real where an OpenCL
device is present.

## 7. Closed in the third pass

Found by diffing the ONNX exporter's op map against the importer's, and by
auditing every public options struct for fields nothing reads.

- **`MaxPool` / `AveragePool` imported as GLOBAL pools** (`src/core/onnx_ops.c`).
  Both handlers reduced over the entire spatial extent, which is
  `GlobalMaxPool`, not a window: `MaxPool(kernel=2, stride=2)` on `[1,1,4,4]`
  returned `[1,1,1,1]` instead of `[1,1,2,2]`, and every downstream layer then
  saw the wrong shape. `op_avgpool` never even read `kernel_shape`, and
  `op_maxpool` returned the input *unchanged* when the input wasn't 4-D or the
  attribute was missing — a silent identity. Both now build `Pool2DParams` from
  `kernel_shape`/`strides`/`pads`/`dilations`/`ceil_mode`/`count_include_pad` and
  call the real `uop_maxpool2d`/`uop_avgpool2d`, refusing (loudly) the cases
  `Pool2DParams` cannot express rather than pooling something else.

- **Nine ops the exporter emits could not be imported** (`src/core/onnx_ops.c`).
  The file header names export→import→run round-tripping as the validation
  strategy, but `Sin`, `Cos`, `Less`, `Greater`, `Equal`, `LessOrEqual`,
  `GreaterOrEqual` and `HardSigmoid` had no importer handler, so a CML-exported
  model failed to load back with "unsupported op". All are now wired to their
  existing uops. `HardSigmoid` refuses non-default `alpha`/`beta` because
  `uop_hard_sigmoid` hardcodes ONNX's 0.2/0.5.

- **`Conv` ignored `auto_pad`** (`src/core/onnx_ops.c`). Models exported from
  TensorFlow carry `auto_pad="SAME_UPPER"` and omit `pads`, so padding silently
  became 0 and the output shape was wrong. `auto_pad` is now honoured
  (`VALID` and the `SAME_*` pair), and both `Conv` and the pooling ops now reject
  asymmetric padding instead of silently keeping only the begin pads — the params
  structs carry one pad per axis and cannot represent it.

- **Declared opset was 11 but the exporter emits opset-12 ops**
  (`src/core/onnx_export.c`). `LessOrEqual`/`GreaterOrEqual` arrived in 12, so
  strict external loaders would reject the models. Now declares 12.

- **`optimizer_set_lr_scheduler()` was a no-op** (`src/optim.c`). It stored
  `step_size`/`gamma` and nothing ever read them, while
  `optimizer_supports_lr_scheduling()` returned true — so the caller was told the
  schedule was active as the learning rate sat still. `optimizer_step()` now
  applies the StepLR decay. (The separate `LRScheduler` subsystem, with its eight
  policies, was always real; this was a vestigial duplicate.)

- **Three dead config knobs removed.** `Optimizer.use_amp` (plus
  `optimizer_set_amp()`), `Optimizer.lr_scheduler_factor`, and
  `CMLScheduleOptions.topological_sort` were written and never read — the same
  shape of bug as §5's `gradient_as_bucket_view`. Mixed precision is the separate
  autocast subsystem; a knob that silently does nothing is worse than no knob.
  The one test touching `topological_sort` only asserted its default was set,
  which verified nothing about behaviour.

- **The 58 example programs were built but never run.** `cml_add_example` only
  created an executable, so a tutorial could crash, or stop compiling against its
  own API, and CI would stay green — and the tutorials are the surface users copy
  from. The fast, self-contained ones now run as `example_*` smoke tests (~31s).
  Deliberately excluded: `mnist_example` (needs a downloaded dataset, and fails
  with a clear message), `gru_classifier` (correct but ~110s), the
  benchmark/profile targets (timing harnesses, not assertions), and the zoo demos
  (already covered by `test_zoo_models`). All 58 were verified to run before
  choosing the subset.

Coverage added: pooling, comparison and round-trip-op cases in
`test_onnx_import_ops.c`; a MaxPool export round-trip in `test_onnx_export.c`;
built-in StepLR cases in `test_optim.c`; 25 example smoke tests.

Two things that looked like gaps and were not, recorded so they are not
re-investigated:

- The exporter's `MaxPool`/`AveragePool` rows are **not** dead code. Exporting a
  pooling graph works and carries every attribute. Export fails only if the graph
  has already been *executed*, because the decompose pass rewrites `MAXPOOL2D`
  into `UNFOLD` + a reduce and `UNFOLD` has no ONNX equivalent. Export the graph
  you built, not the lowered one — the same reason a partial `Flatten` survives to
  the exporter in one graph and arrives pre-lowered in another.
- Comparison ops return **BOOL** tensors (`uop_binary_ex(..., BINARY_DTYPE_BOOL)`),
  one byte per element. Reading that payload as `float32` yields denormals, not
  0/1 — the same trap as the INT64 operands in §6.

## 8. Closed in the fourth pass

The theme this round: the *eager* autodiff engine had gradient rules the graph
engine already had, and nothing ran the gradient suites under it, so the gaps were
invisible. Forcing `GRAD_MODE=eager` on `tests/test_autodiff_ops.c` failed 8 of
its 50 cases while the default run was green.

- **Eager VJPs that graph mode already had** (`src/ops/ir/backward.c`):
  - `DIAGONAL` was grouped with the non-differentiable ops. Its VJP just scatters
    the output gradient back onto the diagonal, and `autodiff.c` had that rule —
    so the same model trained under graph mode and silently didn't under eager.
  - `FLOOR` had no case at all (it fell through to `default`), and `CEIL`,
    `ROUND`, `SIGN`, `TRUNC`, the comparisons and the boolean reducers returned
    *nothing* where the rule is to return a *zero*. Leaving the input's grad NULL
    breaks the chain, so every parameter upstream of a `floor()` stopped training.
    They now call `ensure_grad`, which terminates the chain with zeros.
  - `SORT` and `TOPK` are permutations (TOPK also drops), so their VJP sends each
    output gradient back where it came from. The node stores only
    (dim, descending), but sorting is deterministic: the backward now recomputes
    the ordering with the forward's own helper, so ties resolve identically.
    `lanes_of`/`lane_order` were made non-static (`cml_lanes_of`/`cml_lane_order`)
    for exactly that reason — a second implementation of the comparator would be
    free to disagree with the first.
  - `GATHER` read its index operand through a `float*`. Indices are usually
    INT32/INT64, and a small int read as float32 is a denormal ≈ 0, so every
    gather collapsed onto row 0 and embedding gradients landed on the wrong rows.
    Now routed through `tensor_get_float`, which dispatches on dtype. (Same class
    of bug as the ONNX operands in §6 — worth grepping for as a pattern.)

  - `MASKED_SELECT` was in the same group, and did not belong there. It compacts
    the set positions in order, so its VJP hands element k of the output gradient
    to the k-th set position -- and the mask is `inputs[1]`, already to hand. The
    graph engine had this rule all along. Implemented, with a value check that
    runs in whichever engine `GRAD_MODE` selects.

  `IDIV` and `MOD` remain genuinely non-differentiable. `SPLIT`, `CHUNK` and
  `MESHGRID` also yield no gradient, but the earlier claim here that they "need
  multi-output backward infrastructure" was wrong in a more basic way: **no
  builder anywhere creates those nodes.** They are enum values with no producer
  (ONNX `Split` imports as slices, not `UOP_SPLIT`), so the backward case guards
  an op that cannot occur -- not a gradient gap. Their VJPs would only be worth
  writing alongside an op that actually emits them.

- **Module buffers were dropped by BOTH serialization formats**
  (`src/core/serialization.c`, `src/core/safetensors.c`). Neither walked
  `num_buffers`, so a trained BatchNorm saved and reloaded came back with
  running_mean 0 / running_var 1 and eval-mode inference normalized against the
  *fresh* module's statistics — wrong predictions from a save/load that reported
  success. The native format gained a version-2 buffer section (version 1 files
  still load, they simply carry none); safetensors writes them as extra named
  tensors under a `buffers.` prefix, so older files just lack the keys.
  `copy_into_tensor` also now realizes the destination first — a lazy target has
  `data == NULL`, which silently skipped the copy. Tests for both formats in
  `test_serialization.c`.

- **BatchNorm `momentum` was inverted** (`src/nn/layers/norm.c`). The update put
  the weight on the *old* value, so the documented "typical: 0.1" behaved like
  PyTorch's 0.9 and the running stats tracked little more than the last batch.
  Now `running = (1 - momentum) * running + momentum * batch`, matching the docs
  and every framework a model would be ported from.

- **Negative reduction axes silently reduced everything**
  (`src/autograd/forward_ops.c`). The guard `dim < a->ndim ? dim : -1` tested only
  the upper bound, so any negative axis fell through to `-1`, which downstream is
  the sentinel for reduce-all: `sum(x, -2)` on a `[2,3]` tensor returned a scalar
  instead of reducing axis 0. Negative axes now count from the end. `-1` itself
  stays reduce-all — `zoo/clip.c` and `torch/pte.c` use it that way — and the
  header now says so, because it differs from PyTorch and the surprise should at
  least be written down.

- **`cml_quantize_uint8` silently clamped half its range**
  (`src/core/quantization.c`). `cml_quantize_compute_params(t, false)` bakes in the
  *int8* convention (zero-point offset by -128), and passing that to the uint8
  quantizer — the natural pairing of two public functions — pushed the bottom half
  of the range to 0. Added `cml_quantize_compute_params_uint8()` as the correct
  constructor, and `cml_quantize_uint8` now rejects a zero-point outside [0,255]
  instead of producing quietly wrong output.

- **ConvTranspose could not round-trip** (`src/core/onnx_export.c`,
  `src/core/onnx_ops.c`). The exporter mapped it to an ONNX op name but emitted
  **no attributes at all**, so a loader fell back to defaults and computed
  something else; the importer had no handler either. Both sides now carry
  kernel_shape/strides/dilations/pads/output_padding, and grouped or
  asymmetrically-padded variants are refused rather than approximated.

- **Two more dead knobs removed**: `Module.buffers_populated`
  (`include/nn/layers/sequential.h`) was declared and referenced nowhere; and
  `python/cml/functional.py`'s `LearningRateScheduler` silently did nothing for
  any schedule name other than `"step"`/`"exponential"` — it now raises and points
  at the richer C-backed schedulers in `cml.optim`.

### The "eager nondeterminism" was a test bug, not an engine bug

Worth recording because the first diagnosis was wrong. Forcing
`GRAD_MODE=eager` on `test_autodiff_ops` gave a different score every run
(37/50, 50/50, 37/50 ...) while graph mode was a stable 50/50, and the failing
set was always the same 13 composite unary ops at element `i == 2`. ASAN and LSAN
were clean, which pointed at an uninitialised read somewhere in the eager engine.

It was in the test. `grad_matches`/`forward_sum` built their input with

```c
int sh[2] = {g_rows, g_cols};
Tensor* x = tensor_zeros(sh, g_rows > 1 ? 2 : 1, &cfg);
```

For the elementwise cases `g_rows == 1`, so ndim came out 1 against a `{1, 6}`
shape array — a tensor of shape `{1}`, **one element** — while the callers then
wrote and read `g_rows * g_cols == 6`. Consequences:

- Every `ELEMENTWISE` case validated **element 0 only**. The other five
  comparisons passed vacuously: the numeric derivative of an element the tensor
  does not contain is 0, and the over-read of `x->grad` usually landed on zeroed
  heap, so both sides read 0 and "matched".
- Under the eager engine the over-read sometimes landed on dirty heap instead,
  which is what turned a permanently weak suite into a flaky one. The engine was
  never at fault.

`make_test_input()` now builds `{g_cols}` for the 1-D case and asserts the
tensor's numel equals the element count the caller is about to use, so this
cannot silently come back. Valgrind reported 35 uninitialised-value errors before
and **0** after; eager is 50/50 across 12 consecutive runs, so the gradient
suites are now gated under `GRAD_MODE=eager` in CMakeLists as well.

Two lessons this one is worth keeping for: a passing gradient check can be
vacuous (compare the tensor's real numel against what the test thinks it has),
and ASAN silence is not evidence of memory correctness — valgrind found in one
run what ASAN could not see at all.

## 9. GPU codegen is now numerically validated without a GPU

This was listed for several passes as "codegen exists but numeric validation needs
NVIDIA/AMD hardware (none in CI)". That framing was wrong: it needed *an
execution engine*, and the host can be one.

`test_ptx_codegen.c` asserts the emitted PTX **text** contains the instructions it
should. That catches a missing op but not a wrong one — a kernel that reads the
wrong register, swaps two operands, or picks the wrong rounding mode emits
perfectly plausible text. So the codegen's arithmetic was unverified.

`tests/ptx_interp.h` is a ~500-line interpreter for exactly the PTX subset this
codegen emits, and `tests/test_ptx_numeric.c` runs every generated kernel through
it and compares against the same reference the CPU path computes: **28 unary and
12 binary ops, 43 assertions, no GPU.**

Why it is tractable: the emitted elementwise kernels are straight-line scalar f32
— a thread-index preamble, one predicated `@%pN ret;` bounds guard, computed
`ld.global.f32`/`st.global.f32`, and arithmetic in between. No loops, no shared
memory, no real branching. About 30 distinct opcodes.

Deliberate properties:

- **Addresses are synthetic.** A u64 holds `(buffer_index << 40) | byte_offset`,
  so the pointer arithmetic the kernel does on a param base decodes back to a real
  host array with no flat address space, and an out-of-range access is a reported
  error rather than a host segfault.
- **Unknown instructions fail loudly.** A quiet skip would rebuild exactly the
  false confidence this replaces, so anything outside the supported subset is an
  error — a codegen change that starts emitting new instructions makes the suite
  fail rather than silently stop checking. There is a test asserting this.
- **The bounds guard is tested for real.** `N = 37` with `BLOCK = 8` means 40
  threads for 37 elements, and the buffers are declared oversized so a missing or
  inverted `setp.ge.u32` shows up as a modified tail element — a class of bug that
  is a memory fault on a real GPU and invisible to text inspection.
- **Outputs are poisoned before the run** (`-12345.0f`), so a kernel that writes
  nothing fails instead of passing on leftover zeros.

Verified to be capable of failing, which is the point: injecting a realistic
defect into the codegen (`UOP_SUB` emitting `add.f32`) turns the suite red with a
precise diagnosis —

```
sub    (i=0 a=-1.94595 b=-1.94595 got=-3.89189 want=0)   [FAIL]
Results: 42/43 passed
```

— and reverting restores 43/43. The interpreter is test-only and is not part of
the shipped library. The remaining GPU item is the *driver* path (NVRTC, module
load, launch), which is a genuinely different claim; see §10.

## 10. Deliberately still open (engine-scale or platform-gated)

These are honest, loudly-failing stubs / fallbacks rather than silent fakes.
Each is a cross-cutting engine change with real correctness risk, or needs
hardware not present, so forcing code here would be worse than the honest
fallback:

- **JIT kernel recording** (`src/ops/ir/tiny_jit.c`) — attempted, and the
  original assessment holds; recording it as a trace of IR nodes cannot work. Worth
  writing down so it is not re-attempted the same way.

  `cml_trace_get_active` and `cml_trace_record_kernel` have no call sites outside
  `trace.c`, so no trace ever captured anything and the JIT could never engage.
  The obvious CPU fix looks easy: `cpu_execute_node()` exists, so record the node
  order and replay by calling it — skipping the graph walk, DCE, fusion decisions
  and scheduling without a second engine. Implemented, it records fine (a
  three-op graph yielded one entry after fusion). Replay still cannot work:

  1. **Re-running an executed graph is a no-op.** Nodes are marked `is_executed`,
     so the second `cml_tinyjit_execute` on the same graph recorded *zero* entries —
     there is nothing left to replay.
  2. **Getting a fresh run means `cml_reset_ir_context()`,** which frees the graph.
     Any recorded `IRNode*` is then dangling, so caching one is a use-after-free
     rather than a feature.

  A CPU trace therefore has to record something that outlives the graph — ops plus
  buffer addresses plus shapes, with its own executor over that list. That is
  precisely the "second, replay-shaped CPU engine" this entry always named. The
  node-pointer version was reverted rather than shipped, because a latent
  use-after-free to satisfy a checkbox is worse than an inert feature.

  **What a correct CPU replay would actually require** — the existing design is
  already slot-based and graph-independent, which is the right shape: entries hold
  *slot indices*, and `cml_trace_replay(trace, tensor_ptrs, n)` binds them to the
  current graph's buffers. `cml_ir_output_slots()` enumerates nodes in `head→next`
  order, so slot `i` means the same logical node across two structurally identical
  graphs. Two concrete gaps, not a vague rewrite:

  1. **Slots cover node outputs only.** Graph leaves (inputs and weights) get no
     slot, but every kernel reads them, so a CPU entry cannot name its operands.
     The enumeration has to include leaves.
  2. **Kernels take `IRNode*`, not buffers.** `cpu_execute_node(node)` is the
     dispatcher, so replay needs either a synthesized node built from the record or
     a parallel dispatch keyed on (op type, params, operand buffers, shapes).

  Both are tractable; neither is safe to do casually, because this is the default
  execution path and a wrong field in a synthesized node is silent numerical
  corruption everywhere rather than a crash. Worth doing behind a value-checking
  test (`tests/test_tiny_jit.c` already has `values_ok`) and measured against the
  overhead it is meant to remove — the win is skipping the graph walk, so it should
  be shown to be worth the risk before being switched on.

  Two things found while doing this that matter more than the feature:

  **The empty-trace guard is load-bearing on the DEFAULT path.** TinyJit is on
  unless `TINYJIT=0` (`cml_ir_execute_cpu` dispatches to it), so *every* graph
  execution goes through `cml_tinyjit_execute`. It is harmless today only because
  the CPU trace is always empty and it falls through to real execution. Naively
  "fixing" recording — caching the trace and replaying it — would make every
  repeated graph return its first run's values: silently wrong training, on the
  default path, everywhere. `tests/test_tiny_jit.c` now pins that guard.

  **The codebase already knew.** `cml_ir_reexecute()` — the zero-rebuild
  static-graph step — deliberately calls `cpu_execute_ir` instead of
  `cml_ir_execute`, and says why: *"whose TinyJit replay would return the values
  recorded on the first run instead of recomputing from the updated buffers."* So
  the stale-replay hazard was understood and routed around, not overlooked.

  **Why node pointers can never be the answer**, stated precisely: the trace cache
  lives in a static `g_tinyjit` that outlives any single graph, and it is keyed on a
  *structural* hash — matching structurally identical but **distinct** graph
  instances is the entire point, since that is what makes "build once, replay every
  step" pay off. Holding `IRNode*` into one graph's nodes is fundamentally
  incompatible with that: a freed graph's nodes get replayed as soon as a later
  graph hashes the same. Tying the trace's lifetime to one graph would fix the
  safety and remove the benefit in the same stroke. Hence: record ops plus buffers
  with an executor of their own, which is the second replay-shaped engine.

  Kept from the attempt, since both are real: a `truncated` flag on `CMLTrace`, and
  `cml_trace_end` no longer marking a truncated trace complete. **That was a live
  bug on the GPU path**, not a hypothetical — a trace that hit
  `CML_TRACE_MAX_ENTRIES` silently dropped the overflow, was still marked complete,
  and so was cached and replayed, skipping the graph's tail. `tests/test_tiny_jit.c`
  covers it, along with the fallback itself: results stay correct, and an empty or
  truncated trace is never cached or replayed. That test is what makes the inert
  fallback safe to leave alone — the tempting "fix" of caching the trace anyway
  returns the recording run's values forever.
- **HEVC intra-frame decode** (`src/core/hevc.c`) — NAL and SPS parsing only.
  Previously dismissed here as "out of scope", which was the wrong reason. It needs
  no special hardware, and a verification path does exist on a normal machine:

  ```bash
  ffmpeg -f lavfi -i testsrc2=size=64x64:rate=1:duration=1 -pix_fmt yuv420p \
         -frames:v 1 src.yuv -y
  x265 --input src.yuv --input-res 64x64 --fps 1 --frames 1 --output t.265 \
       --no-deblock --no-sao --keyint 1 --qp 30        # in-loop filters off
  ffmpeg -i t.265 -pix_fmt yuv420p ref.yuv -y          # ground truth
  ```

  What actually blocks it is **reference data, not effort**. A decoder needs the
  spec's exact integer tables — the DCT-II 4/8/16/32 and DST-VII matrices, ~200
  CABAC context-init values, `intraPredAngle`, the scan orders — and a single wrong
  entry produces a decoder that compiles, self-consistently round-trips, and
  decodes real streams wrongly. That is the exact silent-failure mode the rest of
  this document is about removing.

  Reconstructing those tables from memory is demonstrably not safe. The 8-, 16- and
  32-point matrices *can* be derived (`round(64·√2·cos(π·i·(2j+1)/64))` reproduces
  their spec rows exactly), but the 4-point matrix is a hand-adjusted special case:
  the formula yields `[84, 35, -35, -84]` where the spec requires
  `[83, 36, -36, -83]`. One checkable row already disagrees, so the ~200 CABAC
  values — which have no generating formula and no cheap self-check — cannot be
  trusted from recall either.

  Nothing local supplies them: no spec document, no libde265 or HM source, and
  libavcodec ships headers only (the tables live in `.c` files that are not
  installed). So this is open pending the reference tables, at which point the
  verification harness above makes it a tractable, testable project. The honest
  `CM_NOT_IMPLEMENTED` stays until then.

  Fixed along the way, since it blocked any future decoder: `cml_hevc_next_nal()`
  never emitted the **trailing** NAL — a NAL is delimited by the start code that
  follows it, so for a whole-file feed the slice (the only NAL a decoder consumes)
  was permanently unreachable. `cml_hevc_parser_end_of_stream()` is how the caller
  now says no more bytes are coming.
- **The CUDA/ROCm *driver* path** — NVRTC compilation, module load and kernel
  launch still need real hardware. The driver mocks (`test_nv_mock`,
  `test_am_mock`, `test_hip_mock`) cover the transport around it — h2d/d2h,
  submission, ordering — but with a deliberately fake kernel that performs no
  arithmetic, so they say nothing about what a kernel computes.

  The *numeric* half of this item is now closed, and it did not need hardware
  after all. See §9.
- **Non-blocking CI legs** — the macOS matrix leg (`ci.yml`) and the Windows
  wheel build (`wheels.yml`) are `continue-on-error: true`. Promoting either needs
  a green run on that platform first, and this checkout cannot produce one: not
  even a compile check is possible.

Checked rather than assumed, on the machine this work was done on:

| Resource | State |
| --- | --- |
| `/dev/nvidia*`, `/dev/kfd` | absent |
| `libcuda.so[.1]`, `libamdhip64.so`, `libhsa-runtime64.so` | absent |
| `x86_64-w64-mingw32-gcc`, `i686-w64-mingw32-gcc` | absent |
| `clang-cl` | present, but `#include <windows.h>` fails — no Windows SDK |
| osxcross / `o64-clang` | absent |

So each item above is blocked on a resource, not on effort. Anyone with the
matching hardware or toolchain can pick them up directly.

---

## Build & Test

```bash
cmake -S . -B build && make -C build -j$(nproc) && (cd build && ctest --output-on-failure)
```

All 203 ctest suites pass (176 unit suites + 25 example smoke tests + 2
eager-engine gradient re-runs), plus
78 Python tests.

The Python bindings are a separate build (CI runs both):

```bash
cd python && python cml/build_cffi.py && python -m pytest tests/ -q
```

Changing the cffi `cdef` in `python/cml/_cml_cffi.py` requires re-running
`build_cffi.py` — the extension is compiled, not interpreted.
