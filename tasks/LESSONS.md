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
