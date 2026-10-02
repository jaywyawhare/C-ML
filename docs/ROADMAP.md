# C-ML Roadmap: from "serious from-scratch ML library" to framework contender

This is the honest gap list between C-ML today and PyTorch / TensorFlow / tinygrad.
It is deliberately split by **what blocks each item**: effort (can be coded and
verified here), hardware (needs a GPU / Mac this environment lacks), or ecosystem
(adoption and data, not code). Status reflects the tree at the time of writing.

## Where C-ML already stands (not gaps)

Architecturally this is already in tinygrad's weight class and broader than a
from-scratch project usually is:

- UOp IR + decompose/fuse + schedule, LLVM ORC JIT, AOT compile.
- GPU codegen for PTX, SPIR-V, WGSL, Metal, OpenCL, Vulkan (code-complete,
  numerically validated for PTX via an in-repo interpreter — see §Hardware).
- Autograd: eager + graph engines; double-backward / `create_graph` in graph mode.
- Distributed: DDP (bucketed all-reduce, `gradient_as_bucket_view`,
  `find_unused_parameters`), pipeline parallel with GPipe and 1F1B schedules.
- Formats: ONNX import/export, GGUF, SafeTensors.
- Model zoo: ResNet/GPT-2/BERT/ViT/CLIP/ConvNeXt/RetinaNet/Mask R-CNN, etc.
- Python bindings (cffi), AMP/bf16 training, quantization/QAT, 185 passing ctests
  + a Python suite, wheels for Linux x86_64/aarch64 + macOS (Windows best-effort).

## Tier 1 — makes it genuinely competitive (effort-blocked, doable here)

1. **Published performance numbers.** No head-to-head step-time/memory vs torch
   or tinygrad exists. A CPU benchmark is producible here (torch 2.11 CPU is in
   `benchmarks/.venv`); GPU numbers need §Hardware. *Highest-leverage artifact.*
2. **Numerical parity at scale.** A torch-parity harness exists; "contender"
   needs thousands of op/shape/dtype cases checked bit-close against torch in CI,
   not a sample.
3. **Python API breadth.** The cffi `cdef` exposes ~317 of ~1503 public `cml_*`
   functions (~21%). The ergonomic torch-like surface is mostly covered (parity
   tests pass); the gap is long-tail ops and NumPy-grade broadcasting everywhere.
4. **Training-stack depth.** AMP/bf16 proven; still want FSDP/ZeRO-style sharding,
   robust grad-accumulation + loss-scaling at scale, and a fully exercised
   optimizer/scheduler set.

## Tier 2 — the moat (mostly ecosystem, not code)

5. **Pretrained weights out of the box.** GGUF/SafeTensors/ONNX import is the
   foundation; the goal is `llama.generate(...)` / `resnet50(pretrained=True)`.
6. **Real models running end-to-end, fast.** tinygrad's proof is Llama/SD/Whisper.
   C-ML has the architectures; it needs them running with real weights at
   competitive speed (gated on §Hardware for the "fast" half).
7. **Deployment story.** The WebGPU + AOT paths could become a genuine
   browser/edge advantage (TF's turf) — unproven, needs an end-to-end demo.
8. **Docs, tutorials, installable packages, community.** Years-long, social.

## Hardware-blocked (no code to write here — needs a device)

- **Real GPU execution + benchmarks.** Every GPU backend is code-complete and
  mock/PTX-interpreter-validated, but nothing has run on NVIDIA/AMD/Apple silicon.
  This is the #1 blocker to any "killer" claim. Resource check on this machine:
  `/dev/nvidia*`, `/dev/kfd`, `libcuda`, `libamdhip64`, `libhsa-runtime64` all
  absent. Driver code (NVRTC/HIP module load+launch) is implemented and
  mock-tested; it needs a run on a real device, not more code.
- **macOS CI leg** — needs a real Mac (no osxcross / Darwin SDK).

## Data/spec-blocked (needs external reference material)

- **HEVC intra-frame decode** — needs the spec's exact integer tables (DCT/DST
  matrices, ~200 CABAC context-init values, scan orders). Not reconstructable
  locally; a single wrong entry silently mis-decodes. Verification harness is
  documented in REMAINING_WORK.md §10.

## The single highest-leverage next move

Get it running on **one real GPU**, train **one real model** (GPT-2 small or
ResNet-50) end-to-end, and publish **step-time + memory vs torch and tinygrad**
on that device. That one artifact converts "impressive from-scratch project" into
"credible contender" faster than any number of additional features — and it is the
one thing this environment cannot produce (no GPU).
