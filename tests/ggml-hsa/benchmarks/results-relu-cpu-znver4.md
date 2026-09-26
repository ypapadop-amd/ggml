# bench-relu-hsa — RELU CPU (znver4)

`out = relu(a)` (element-wise, memory-bound) via `GGML_UNARY_OP_RELU`, from `results-relu-cpu-znver4.json`.

## Environment

| | |
|---|---|
| Host | aus-mini-ypapadop |
| CPUs | 16 |
| CPU MHz | 5054 |
| Harness build | release |
| Timestamp | 2026-07-20T17:39:39-04:00 |
| Last-level cache | 16384 KiB (L3) |

## Results

### F32

| shape (ne0×ne1×ne2×ne3) | elements | time (µs) | stddev (µs) | GB/s | reps × iters |
|---|---:|---:|---:|---:|---:|
| 500×500×1×1 | 250000 | 19.02 | 0.62 | 105.3 | 5 × 22178 |
| 14×14×16×500 | 1568000 | 1072.65 | 3.07 | 11.7 | 5 × 388 |
| 28×28×8×500 | 3136000 | 1072.49 | 6.04 | 23.4 | 5 × 396 |
