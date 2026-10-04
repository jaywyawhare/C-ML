# Benchmarks

Head-to-head timing against PyTorch. **CPU only** — this environment has no GPU,
so GPU numbers (the ones that matter most for a framework claim) are still owed;
These are a baseline, not a victory lap.

## Method

`benchmarks/bench_vs_torch_cpu.py`. Each workload is realized every iteration
(C-ML is lazy — `.numpy()` forces execution; torch eager realizes on `.numpy()`
too), warmed up 5 runs, then timed as the **median of 30**. Ratio < 1.0 means
C-ML is faster. Numbers are host-specific; re-run locally to reproduce:

```bash
cd python && ../benchmarks/.venv/bin/python cml/build_cffi.py   # build binding
PYTHONPATH=python benchmarks/.venv/bin/python benchmarks/bench_vs_torch_cpu.py
```

## Representative run (PyTorch 2.11.0+cpu, 10 threads)

| workload            | C-ML ms | torch ms | ratio  | note         |
| ------------------- | ------- | -------- | ------ | ------------ |
| matmul 64x64        |   0.057 |    0.053 | 1.07x  | torch faster |
| matmul 256x256      |   0.585 |    1.936 | 0.30x  | **C-ML faster** |
| matmul 512x512      |   3.843 |    3.897 | 0.99x  | ~tie         |
| matmul 1024x1024    |  25.566 |   21.352 | 1.20x  | torch faster |
| elementwise 4096    |   0.083 |    0.019 | 4.37x  | torch faster |
| elementwise 65536   |   0.110 |    0.083 | 1.32x  | torch faster |
| elementwise 1048576 |   2.232 |    3.030 | 0.74x  | **C-ML faster** |

## Reading it honestly

- **Large GEMM is a BLAS wash** — both call optimized BLAS, so 512²–1024² land
  within ~20% of each other. Expected.
- **C-ML wins mid-size matmul and large elementwise**, where its fusion + lower
  per-op overhead pay off against torch's dispatch.
- **C-ML loses on tiny tensors** (elementwise 4096): fixed per-op Python/binding
  overhead dominates when there's almost no arithmetic. This is the clearest
  CPU-side optimization target.

Being roughly on par with PyTorch on CPU from a from-scratch C library is a real
result; the open question a framework claim hinges on is GPU throughput, which
cannot be measured here.
