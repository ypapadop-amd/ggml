# Lessons — hsa-gpu-native-reuse (ggml-cuda HIP kernels on native ggml-hsa)

Redo of the HIP-shim work, restructured to reuse NATIVE ggml-hsa machinery instead of a
duplicated shim. Branch: hsa-gpu-native-reuse (ggml worktree) + hsa-native-reuse (llama.cpp_gold).

## NR1 — Architecture: extend 3 native functions + one thin glue file
- kernel-discovery.cpp: factored the code-object load/freeze + symbol-info query into shared
  helpers `ggml_hsa_load_executable` / `ggml_hsa_resolve_kernel_symbol` / `ggml_hsa_load_code_object`
  (kernel-discovery.hpp). Native `ggml_hsa_create_gpu_kernel` calls them (no behavior change); the
  HIP ABI calls load_executable once per fatbin then resolve per-kernel.
- hsaco-metadata.{hpp,cpp}: the neutral multi-kernel comgr parser (`ggml_hsaco_parse_kernels`).
  Native single-kernel `ggml_hsa_parse_kernel_metadata` is now a thin wrapper over it.
- gpu-kernel.cpp: `ggml_hsa_gpu_dispatch_generic` beside `ggml_hsa_gpu_kernel::dispatch`, reusing
  the native kernarg pool alloc/align, unified-access grant, queue reserve, per-packet completion
  signal on ctx.dispatch_signal, and doorbell verbatim. New = positional void** packing + COV5
  hidden args from a HIP grid/block.
- hip-abi.cpp: the ~50 hip*/__hip* symbols routed onto the above. hip_shim_blas.cpp reused as-is.
- ggml-hsa.cpp: GGML_HSA_GPU_DELEGATE block (delegate GPU device to ggml-cuda) — identical logic to
  the shim's proven delegation, now with ggml-cuda compiled INTO libggml-hsa.

## NR2 — SYNCHRONOUS v1 with the native counting signal works and is coherent
hipLaunchKernel = dispatch_generic (completion_signal = ctx.dispatch_signal, add 1, doorbell) then
`ggml_hsa_wait_dispatches` immediately. Only one packet in flight at a time, so the signal never
goes negative (the L15 async hazard does not apply to the synchronous model). -fa on, -ngl 99,
Llama-3.1-8B Q4_K_M = "The capital of France is<|im_end|>" (matches build-realhip). No async /
barrier-bit needed for correctness; async is a later optimization.

## NR3 — CMake: compile ggml-cuda INTO libggml-hsa; do NOT link hip::device (CRITICAL)
The new architecture compiles ../ggml-cuda/*.cu with hipcc into the ggml-hsa library (no ggml-hip
target -> no circular dep: hip-abi calls native dispatch, ggml-hsa delegates to ggml-cuda).
- `enable_language(HIP)`, set the .cu LANGUAGE HIP (they get --offload-arch from
  CMAKE_HIP_ARCHITECTURES automatically); set hip-abi.cpp / hip_shim_blas.cpp / hsaco-metadata.cpp
  LANGUAGE CXX with `__HIP_PLATFORM_AMD__`.
- **Do NOT `target_link_libraries(... hip::device)`** — its INTERFACE options add `--offload-arch`,
  which is applied to the plain-C++ TUs in the same target and g++ rejects it
  (`unrecognized command-line option '--offload-arch=gfx1151'`). The HIP compiler supplies the
  device libs itself. Link only ggml-base/hsa-runtime64/amd_comgr/BLAS + `-Wl,-Bsymbolic-functions`
  (so hip-abi's hip* win over the implicitly-NEEDED libamdhip64), LINKER_LANGUAGE CXX.
- Gate on a distinct GGML_HSA_GPU_DELEGATE so the native Triton GPU build (GGML_HSA_GPU only) is
  unaffected.

## NR4 — llama.cpp_gold mirror is version-safe
Native worktree is ggml 0.15.3; llama.cpp_gold is 0.16.0. common.hpp/host-ops.cpp are byte-identical
across the two, and ggml-hsa.cpp differs only in comments -> the ggml-hsa backend is
version-independent between 0.15.3 and 0.16.0 (confirms the fork's clean rebase). Mirror =
overwrite ggml/src/ggml-hsa/* with the native worktree's, build with GGML_HIP=OFF GGML_HSA=ON
GGML_HSA_GPU=ON GGML_HSA_GPU_DELEGATE=ON GGML_HSA_AIE=OFF GGML_HSA_JIT_COMPILE=OFF. Build dir:
build-hsa-native (build-realhip / build-hsa-be kept as references).

---

# Lessons — hsa-gpu-gemm (GPU GEMM replacing the CPU hipBLAS shim)

Branch hsa-gpu-gemm (ggml worktree) + hsa-gemm (llama.cpp_gold). Goal: replace the correctness-first
host CPU sgemm_ behind the hipBLAS entry points with a real GPU GEMM on gfx1151.

## GG1 — Design: a hipcc .cu whose <<<>>> funnels through the existing HIP ABI
- src/ggml-hsa/hip_gemm.cu: one tiled (TS=16) shared-memory GEMM kernel templated on <TA,TB,TC>
  (float/__half/bf16), column-major C = alpha*op(A)*op(B) + beta*C, F32 accumulate. BF16 carried as
  raw uint16 with in-device bit conversion (no hip_bf16 header dep); F16 via __half + hip_fp16.h.
  A single extern "C" ggml_hsa_hip_gemm(opA,opB,m,n,k,alpha,A,tA,lda,B,tB,ldb,beta,C,tC,ldc) does a
  3x3x3 nested switch on the hipDataType codes -> template launch. No new dispatch machinery: the
  <<<>>> lowers to __hipPushCallConfiguration + hipLaunchKernel which hip-abi.cpp already routes onto
  native ggml_hsa_gpu_dispatch_generic (synchronous, drains the queue). It auto-registers via the
  fatbin __hipRegisterFatBinary path like every ported ggml-cuda TU.
- REUSE hip_shim_blas.cpp verbatim for arg interpretation (transpose N/T, alpha/beta by
  hipblasComputeType_t, F16/BF16 element sizes, strided + pointer-array batched loops). Only gemm_one's
  inner compute changed: it now hands the RAW unified-memory device pointers straight to
  ggml_hsa_hip_gemm — no host F32 conversion, no round-trip. Batched variants get GPU for free (each
  batch = one launch; the existing loops are untouched). GGML_HSA_GEMM_CPU=1 keeps the sgemm_ path.

## GG2 — CMake: add the .cu to GGML_SOURCES_CUDA (one line)
hip_gemm.cu lives in src/ggml-hsa (not ../ggml-cuda) so the glob misses it; `list(APPEND
GGML_SOURCES_CUDA hip_gemm.cu)` inside the DELEGATE block is enough — it then inherits LANGUAGE HIP +
--offload-arch from the existing set_source_files_properties. No other build changes; libblas link
stays (CPU fallback). Do NOT link hip::device (NR3 still applies).

## GG3 — wave32, correctness, before/after
- TS=16 -> 256 threads/block = 8 wave32 waves; no MFMA used (gfx1151 has none). Plain FMA tiling.
- Correctness: test-backend-ops MUL_MAT on HSA0 = 1134/1134 PASS (f32/f16/bf16, single + batched
  bs=[8,3]) vs CPU ref; MNIST MUL_MAT 3/3 PASS.
- Perf (MNIST 500x500x784 f32): CPU sgemm_ 5.06 GFLOPS -> GPU 1.39 TFLOPS (~275x). Small/skinny
  GEMMs ~5 -> ~220 GFLOPS. Big f32 (4096x512x14336) 1.23 TFLOPS. Proves GPU execution (CPU can't hit
  TFLOPS). GGML_HSA_GEMM_CPU=1 reproduces the old 5 GFLOPS -> confirms the switch.
- Llama Q4_K_M -ngl 99: -fa on coherent 38.3 t/s (MMQ+FA, does NOT hit the shim -> unregressed vs
  build-realhip); -fa off now coherent 18.4 t/s ("...the city of Paris."), running the batched
  attention GEMM on GPU (previously CPU). -fa off < -fa on because per-head synchronous launches beat
  nothing fancy yet.
- Headroom: TS=16 naive tiling leaves lots on the table (register blocking / larger tiles / vectorized
  loads / batch as grid.z to fuse per-head launches / async pipelining). Correctness-first; tune later.

## GG4 — llama-cli 0.16 hangs after generation on </dev/null (cosmetic)
llama-cli enters an interactive turn loop and spews empty `>` prompts reading EOF from /dev/null even
with -no-cnv; generation completes first (grep the answer + "t/s" line). Wrap in `timeout` and pkill;
the exit 143/144 is the kill, not a failure.
