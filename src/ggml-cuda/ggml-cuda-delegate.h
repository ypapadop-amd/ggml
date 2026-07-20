#pragma once
struct ggml_backend_cuda_context;
struct ggml_tensor;
// Execute a single graph node on `ctx` (device is set internally). Exposed so the ggml-hsa GPU
// backend can own the graph-compute loop while reusing the ggml-cuda op kernels (dispatched on
// the HSA runtime by hip-abi). Returns false if the op is unsupported.
bool ggml_cuda_delegate_forward_op(struct ggml_backend_cuda_context * ctx, struct ggml_tensor * node);
