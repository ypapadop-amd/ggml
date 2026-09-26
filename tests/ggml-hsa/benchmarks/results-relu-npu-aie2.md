# bench-relu-hsa — RELU NPU (HSA, aie2, IRON)

`out = relu(a)` (element-wise, memory-bound) via `GGML_UNARY_OP_RELU`, from `results-relu-npu-aie2.json`.

## Environment

| | |
|---|---|
| Host | aus-mini-ypapadop |
| CPUs | 16 |
| CPU MHz | 3376 |
| Harness build | release |
| Timestamp | 2026-07-20T17:36:18-04:00 |
| Last-level cache | 16384 KiB (L3) |

## Results

### F32

| shape (ne0×ne1×ne2×ne3) | elements | time (µs) | stddev (µs) | GB/s | reps × iters |
|---|---:|---:|---:|---:|---:|
| 500×500×1×1 | 250000 | 602.75 | 2.09 | 3.3 | 5 × 703 |
| 14×14×16×500 | 1568000 | 3070.93 | 10.38 | 4.1 | 5 × 137 |
| 28×28×8×500 | 3136000 | 5970.26 | 8.22 | 4.2 | 5 × 70 |
