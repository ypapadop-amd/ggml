// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// hip-abi: a link-time HIP host runtime that lets the ggml-cuda kernels (compiled by hipcc for
// gfx*) run on the AMD GPU through the NATIVE ggml-hsa backend machinery — WITHOUT libamdhip64.
//
// ggml-cuda's host code funnels every <<<>>> launch through a small set of HIP entry points
// (__hipRegister*, __hip{Push,Pop}CallConfiguration, hipLaunchKernel) plus memory/stream/device
// calls. This file implements exactly those symbols, but — unlike a standalone shim — it does NOT
// stand up its own HSA runtime. It reuses the native ggml-hsa machinery end to end:
//   - device discovery + agents + memory pools: ggml_hsa_info() / ggml_hsa_get_device_info()
//   - queue + completion signal: a native ggml_backend_hsa_context (see native_ctx())
//   - code-object load/freeze + symbol resolve: ggml_hsa_load_executable /
//     ggml_hsa_resolve_kernel_symbol (kernel-discovery.cpp)
//   - multi-kernel metadata parse: ggml_hsaco_parse_kernels (hsaco-metadata.cpp)
//   - AQL kernarg packing + dispatch: ggml_hsa_gpu_dispatch_generic (gpu-kernel.cpp)
//   - synchronisation: ggml_hsa_wait_dispatches (ggml-hsa.cpp)
// The only genuinely new logic here is the HIP ABI surface itself: fatbin bundle unwrap, the
// registration/launch funnel, and the memory/stream/device query wrappers.
//
// Bring-up model is intentionally SYNCHRONOUS (each launch waits) for correctness; async overlap is
// a later optimization. See tasks/LESSONS.md: L1 (registration runs in a global ctor before other
// TUs' static init -> use lazy function-local statics and DEFER all HSA/comgr work to first launch),
// L14 (report wave32 wavefront size from HSA, never hardcode 64).

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "ggml-hsa/common.hpp"
#include "ggml-hsa/gpu-kernel.hpp"
#include "ggml-hsa/hsaco-metadata.hpp"
#include "ggml-hsa/kernel-discovery.hpp"

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// -----------------------------------------------------------------------------------------------
// Native device + context accessors (all lazy: L1 forbids touching HSA before main).
// -----------------------------------------------------------------------------------------------

/// @brief Native ggml-hsa device id of the (first) GPU agent.
int gpu_device_id() {
    static int id = [] {
        const auto & info = ggml_hsa_info();
        for (std::int32_t i = 0; i < info.device_count; ++i) {
            if (info.devices[i].type == HSA_DEVICE_TYPE_GPU) {
                return static_cast<int>(i);
            }
        }
        GGML_ABORT("[hip-abi] no HSA GPU device found\n");
    }();
    return id;
}

const ggml_hsa_device_info::device_info & gpu_dev_info() {
    return ggml_hsa_get_device_info(gpu_device_id());
}

/// @brief The single native HSA context (queue + completion signal) used for all HIP launches.
///
/// Reuses ggml-hsa's own context type — its constructor creates the HSA queue and signal via the
/// native helpers, so no second queue/signal machinery is introduced here.
ggml_backend_hsa_context & native_ctx() {
    static ggml_backend_hsa_context ctx(gpu_dev_info());
    static bool announced = [] {
        fprintf(stderr,
                "[hip-abi] active: ggml-cuda HIP kernels dispatched via native ggml-hsa (no "
                "libamdhip64)\n");
        return true;
    }();
    (void) announced;
    return ctx;
}

/// @brief Grants the GPU and CPU agents access to a unified-memory allocation (APU shared memory).
void grant_unified(void * ptr) {
    const auto & info = ggml_hsa_info();
    hsa_agent_t agents[2];
    std::uint32_t n = 0;
    if (gpu_dev_info().agent.handle != 0) {
        agents[n++] = gpu_dev_info().agent;
    }
    if (info.cpu_agent.handle != 0) {
        agents[n++] = info.cpu_agent;
    }
    if (n != 0) {
        hsa_amd_agents_allow_access(n, agents, nullptr, ptr);
    }
}

/// @brief The GPU's native wavefront size (32 on gfx11xx). Query it — never hardcode (L14).
std::uint32_t gpu_wavefront_size() {
    static std::uint32_t ws = [] {
        std::uint32_t w = 0;
        hsa_agent_get_info(gpu_dev_info().agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &w);
        return w ? w : 32u;
    }();
    return ws;
}

// -----------------------------------------------------------------------------------------------
// Registered modules + kernels (registration is deferred; loading happens on first launch — L1).
// -----------------------------------------------------------------------------------------------

/// @brief A single kernel resolved from a fatbin.
struct KernelInfo {
    std::uint64_t kernel_object = 0;
    std::uint32_t private_seg = 0;
    std::uint32_t group_seg = 0;
    std::uint32_t kernarg_size = 0;
    std::uint32_t kernarg_align = 16;
    std::vector<ggml_hsa_gpu_kernel_arg> args;
};

/// @brief A registered fatbin (one hipcc TU -> many kernels). Loaded lazily on first launch.
struct Module {
    const void * bundle = nullptr; ///< __CLANG_OFFLOAD_BUNDLE__ pointer (FatBinWrapper::binary).
    bool loaded = false;
    hsa_executable_t executable{};
    std::unordered_map<std::string, KernelInfo> kernels; ///< key: mangled name (no ".kd").
};

std::mutex & registry_mutex() {
    static std::mutex m;
    return m;
}

/// @brief Per-host-stub registration: which module + which kernel name it maps to.
struct StubReg {
    Module * module = nullptr;
    std::string name;
    KernelInfo * resolved = nullptr; ///< cached after first lookup
};

std::unordered_map<const void *, StubReg> & stub_registry() {
    static std::unordered_map<const void *, StubReg> m;
    return m;
}

Module *& pending_module() {
    static Module * m = nullptr;
    return m;
}

// thread-local <<<>>> launch configuration (single depth suffices).
thread_local dim3 tls_grid;
thread_local dim3 tls_block;
thread_local std::size_t tls_shmem = 0;
thread_local hipStream_t tls_stream = nullptr;

/// @brief Extracts [ptr,size) of the amdgcn ELF from a __CLANG_OFFLOAD_BUNDLE__ blob.
bool extract_amdgcn_elf(const void * bundle, const void *& elf, std::size_t & elf_size) {
    const char * base = static_cast<const char *>(bundle);
    static const char MAGIC[] = "__CLANG_OFFLOAD_BUNDLE__"; // 24 bytes, no NUL
    if (std::memcmp(base, MAGIC, 24) != 0) {
        return false;
    }
    const char * p = base + 24;
    std::uint64_t num = 0;
    std::memcpy(&num, p, 8);
    p += 8;
    for (std::uint64_t i = 0; i < num; ++i) {
        std::uint64_t off = 0, size = 0, tlen = 0;
        std::memcpy(&off, p, 8);
        p += 8;
        std::memcpy(&size, p, 8);
        p += 8;
        std::memcpy(&tlen, p, 8);
        p += 8;
        std::string triple(p, tlen);
        p += tlen;
        if (triple.find("amdgcn") != std::string::npos) {
            elf = base + off;
            elf_size = size;
            return true;
        }
    }
    return false;
}

/// @brief Loads a module's code object and resolves all of its kernels (once). Caller holds the
///        registry mutex.
void ensure_module_loaded(Module * m) {
    if (m->loaded) {
        return;
    }
    const auto & dev = gpu_dev_info();

    const void * elf = nullptr;
    std::size_t elf_size = 0;
    if (!extract_amdgcn_elf(m->bundle, elf, elf_size)) {
        GGML_ABORT("[hip-abi] fatbin has no amdgcn bundle entry\n");
    }

    if (!ggml_hsa_load_executable(dev, elf, elf_size, m->executable)) {
        GGML_ABORT("[hip-abi] failed to load code object\n");
    }

    std::unordered_map<std::string, ggml_hsaco_kernel> parsed;
    if (!ggml_hsaco_parse_kernels(elf, elf_size, parsed)) {
        GGML_ABORT("[hip-abi] failed to parse kernel metadata\n");
    }

    for (auto & kv : parsed) {
        ggml_hsa_loaded_kernel loaded;
        if (!ggml_hsa_resolve_kernel_symbol(dev, m->executable, kv.second.symbol, loaded)) {
            continue; // not a resolvable kernel symbol (shouldn't happen for amdhsa.kernels entries)
        }
        KernelInfo ki;
        ki.kernel_object = loaded.kernel_object;
        ki.private_seg = loaded.private_segment_size;
        ki.group_seg = loaded.group_segment_size;
        ki.kernarg_size = loaded.kernarg_size;
        ki.kernarg_align = loaded.kernarg_align;
        ki.args.reserve(kv.second.args.size());
        for (const auto & a : kv.second.args) {
            ggml_hsa_gpu_kernel_arg info;
            info.name = a.name;
            info.value_kind = a.value_kind;
            info.offset = a.offset;
            info.size = a.size;
            ki.args.push_back(std::move(info));
        }
        m->kernels.emplace(kv.first, std::move(ki));
    }
    m->loaded = true;
}

} // namespace

// ===============================================================================================
// HIP registration + launch ABI
// ===============================================================================================
extern "C" {

void ** __hipRegisterFatBinary(const void * data) {
    struct FatBinWrapper {
        int magic;
        int version;
        void * binary;
        void * unused;
    };
    const auto * w = static_cast<const FatBinWrapper *>(data);
    // DEFER all HSA/comgr work to first launch (L1): just remember the bundle pointer here.
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto * m = new Module();
    m->bundle = w->binary;
    pending_module() = m;
    static int token = 0;
    return reinterpret_cast<void **>(&token);
}

void __hipRegisterFunction(void ** /*modules*/, const void * hostFunction, char * deviceFunction,
                           const char * deviceName, unsigned int /*threadLimit*/, void * /*tid*/,
                           void * /*bid*/, void * /*blockDim*/, void * /*gridDim*/,
                           int * /*wSize*/) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    Module * m = pending_module();
    if (m == nullptr) {
        return;
    }
    const char * name = deviceFunction ? deviceFunction : deviceName;
    stub_registry()[hostFunction] = StubReg{m, std::string(name ? name : ""), nullptr};
}

void __hipUnregisterFatBinary(void ** /*modules*/) {}

void __hipRegisterVar(void ** /*modules*/, void * /*var*/, char * /*hostVar*/, char * /*deviceVar*/,
                      int /*ext*/, std::size_t /*size*/, int /*constant*/, int /*global*/) {}
void __hipRegisterManagedVar(void * /*module*/, void ** /*pointer*/, void * /*init*/,
                             const char * /*name*/, std::size_t /*size*/, unsigned /*align*/) {}
void __hipRegisterSurface(void ** /*modules*/, void * /*var*/, char * /*hostVar*/,
                          char * /*deviceVar*/, int /*type*/, int /*ext*/) {}
void __hipRegisterTexture(void ** /*modules*/, void * /*var*/, char * /*hostVar*/,
                          char * /*deviceVar*/, int /*type*/, int /*norm*/, int /*ext*/) {}

hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, std::size_t sharedMem,
                                      hipStream_t stream) {
    tls_grid = gridDim;
    tls_block = blockDim;
    tls_shmem = sharedMem;
    tls_stream = stream;
    return hipSuccess;
}

hipError_t __hipPopCallConfiguration(dim3 * gridDim, dim3 * blockDim, std::size_t * sharedMem,
                                     hipStream_t * stream) {
    *gridDim = tls_grid;
    *blockDim = tls_block;
    *sharedMem = tls_shmem;
    *stream = tls_stream;
    return hipSuccess;
}

hipError_t hipLaunchKernel(const void * function_address, dim3 numBlocks, dim3 dimBlocks,
                           void ** args, std::size_t sharedMemBytes, hipStream_t /*stream*/) {
    KernelInfo * ki = nullptr;
    {
        std::lock_guard<std::mutex> lk(registry_mutex());
        auto it = stub_registry().find(function_address);
        if (it == stub_registry().end()) {
            fprintf(stderr, "[hip-abi] hipLaunchKernel: unregistered function %p\n",
                    function_address);
            return hipErrorInvalidDeviceFunction;
        }
        StubReg & reg = it->second;
        if (reg.resolved == nullptr) {
            ensure_module_loaded(reg.module);
            auto kit = reg.module->kernels.find(reg.name);
            if (kit == reg.module->kernels.end()) {
                fprintf(stderr, "[hip-abi] hipLaunchKernel: kernel '%s' not found in module\n",
                        reg.name.c_str());
                return hipErrorInvalidDeviceFunction;
            }
            reg.resolved = &kit->second;
        }
        ki = reg.resolved;
    }

    const ggml_hsa_dim3 grid{numBlocks.x, numBlocks.y, numBlocks.z};
    const ggml_hsa_dim3 block{dimBlocks.x, dimBlocks.y, dimBlocks.z};

    ggml_backend_hsa_context & ctx = native_ctx();
    ggml_status st = ggml_hsa_gpu_dispatch_generic(
        ctx, ki->kernel_object, ki->args, args, grid, block,
        static_cast<std::uint32_t>(sharedMemBytes), ki->private_seg, ki->group_seg, ki->kernarg_size,
        ki->kernarg_align);
    if (st != GGML_STATUS_SUCCESS) {
        return hipErrorLaunchFailure;
    }
    // Synchronous bring-up: wait for the launch to complete (reuses the native drain, which also
    // frees the kernarg allocation tracked by dispatch_generic).
    ggml_hsa_wait_dispatches(ctx);
    return hipSuccess;
}

hipError_t hipLaunchCooperativeKernel(const void * f, dim3 gridDim, dim3 blockDim,
                                      void ** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream) {
    // Bring-up: dispatch like a normal kernel. True grid-wide cooperative semantics are not
    // guaranteed; revisit if a used kernel relies on them.
    return hipLaunchKernel(f, gridDim, blockDim, kernelParams, sharedMemBytes, stream);
}

hipError_t hipFuncSetAttribute(const void * /*func*/, hipFuncAttribute /*attr*/, int /*value*/) {
    // Dynamic-LDS opt-in: kernargs already carry per-launch group_segment_size, so this is a no-op
    // for the ops we run.
    return hipSuccess;
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * /*f*/,
                                                        int /*blockSize*/,
                                                        std::size_t /*dynSharedMemPerBlk*/) {
    if (numBlocks) {
        *numBlocks = 8;
    }
    return hipSuccess;
}

} // extern "C"

// ===============================================================================================
// Memory (unified on APU: allocate from the native data pool + grant CPU/GPU access; copies are
// plain memcpy because host and device share physical memory)
// ===============================================================================================
extern "C" {

hipError_t hipMalloc(void ** ptr, std::size_t size) {
    if (size == 0) {
        size = 1;
    }
    if (hsa_amd_memory_pool_allocate(gpu_dev_info().data_memory.memory_pool, size, 0, ptr) !=
        HSA_STATUS_SUCCESS) {
        return hipErrorOutOfMemory;
    }
    grant_unified(*ptr);
    return hipSuccess;
}

hipError_t hipFree(void * ptr) {
    if (ptr) {
        hsa_amd_memory_pool_free(ptr);
    }
    return hipSuccess;
}

hipError_t hipMallocManaged(void ** ptr, std::size_t size, unsigned int /*flags*/) {
    return hipMalloc(ptr, size);
}
hipError_t hipHostMalloc(void ** ptr, std::size_t size, unsigned int /*flags*/) {
    return hipMalloc(ptr, size);
}
hipError_t hipHostFree(void * ptr) {
    return hipFree(ptr);
}
hipError_t hipHostRegister(void * /*ptr*/, std::size_t /*size*/, unsigned int /*flags*/) {
    return hipSuccess; // unified memory: host pointers are already device-visible
}
hipError_t hipHostUnregister(void * /*ptr*/) {
    return hipSuccess;
}
hipError_t hipHostGetDevicePointer(void ** dev, void * host, unsigned int /*flags*/) {
    *dev = host;
    return hipSuccess;
}

hipError_t hipMemcpy(void * dst, const void * src, std::size_t sizeBytes, hipMemcpyKind /*kind*/) {
    ggml_hsa_wait_dispatches(native_ctx()); // ensure prior kernels finished (unified memory)
    std::memcpy(dst, src, sizeBytes);
    return hipSuccess;
}
hipError_t hipMemcpyAsync(void * dst, const void * src, std::size_t sizeBytes,
                          hipMemcpyKind /*kind*/, hipStream_t /*stream*/) {
    ggml_hsa_wait_dispatches(native_ctx());
    std::memcpy(dst, src, sizeBytes);
    return hipSuccess;
}
hipError_t hipMemcpy2DAsync(void * dst, std::size_t dpitch, const void * src, std::size_t spitch,
                            std::size_t width, std::size_t height, hipMemcpyKind /*kind*/,
                            hipStream_t /*stream*/) {
    ggml_hsa_wait_dispatches(native_ctx());
    for (std::size_t r = 0; r < height; ++r) {
        std::memcpy(static_cast<char *>(dst) + r * dpitch,
                    static_cast<const char *>(src) + r * spitch, width);
    }
    return hipSuccess;
}
hipError_t hipMemcpyPeerAsync(void * dst, int /*dstDev*/, const void * src, int /*srcDev*/,
                              std::size_t sizeBytes, hipStream_t /*stream*/) {
    return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, nullptr);
}
hipError_t hipMemset(void * dst, int value, std::size_t sizeBytes) {
    ggml_hsa_wait_dispatches(native_ctx());
    std::memset(dst, value, sizeBytes);
    return hipSuccess;
}
hipError_t hipMemsetAsync(void * dst, int value, std::size_t sizeBytes, hipStream_t /*stream*/) {
    return hipMemset(dst, value, sizeBytes);
}
hipError_t hipMemAdvise(const void * /*dev*/, std::size_t /*count*/, hipMemoryAdvise /*advice*/,
                        int /*device*/) {
    return hipSuccess;
}
hipError_t hipMemGetInfo(std::size_t * free, std::size_t * total) {
    std::size_t sz = 0;
    hsa_amd_memory_pool_get_info(gpu_dev_info().data_memory.memory_pool,
                                 HSA_AMD_MEMORY_POOL_INFO_SIZE, &sz);
    if (total) {
        *total = sz;
    }
    if (free) {
        *free = sz;
    }
    return hipSuccess;
}

} // extern "C"

// ===============================================================================================
// Stream / event / sync (synchronous bring-up: streams alias the single in-order native queue)
// ===============================================================================================
extern "C" {

hipError_t hipStreamCreateWithFlags(hipStream_t * stream, unsigned int /*flags*/) {
    static int tok = 0;
    *stream = reinterpret_cast<hipStream_t>(&tok + 1); // opaque non-null handle
    return hipSuccess;
}
hipError_t hipStreamCreate(hipStream_t * stream) {
    return hipStreamCreateWithFlags(stream, 0);
}
hipError_t hipStreamCreateWithPriority(hipStream_t * stream, unsigned int flags, int /*priority*/) {
    return hipStreamCreateWithFlags(stream, flags);
}
hipError_t hipStreamDestroy(hipStream_t /*stream*/) {
    return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t /*stream*/) {
    ggml_hsa_wait_dispatches(native_ctx());
    return hipSuccess;
}
hipError_t hipStreamWaitEvent(hipStream_t /*stream*/, hipEvent_t /*event*/, unsigned int /*flags*/) {
    ggml_hsa_wait_dispatches(native_ctx());
    return hipSuccess;
}
hipError_t hipStreamQuery(hipStream_t /*stream*/) {
    return hipSuccess;
}
hipError_t hipDeviceSynchronize() {
    ggml_hsa_wait_dispatches(native_ctx());
    return hipSuccess;
}

hipError_t hipEventCreateWithFlags(hipEvent_t * event, unsigned int /*flags*/) {
    static int tok = 0;
    *event = reinterpret_cast<hipEvent_t>(&tok + 1);
    return hipSuccess;
}
hipError_t hipEventCreate(hipEvent_t * event) {
    return hipEventCreateWithFlags(event, 0);
}
hipError_t hipEventDestroy(hipEvent_t /*event*/) {
    return hipSuccess;
}
hipError_t hipEventRecord(hipEvent_t /*event*/, hipStream_t /*stream*/) {
    ggml_hsa_wait_dispatches(native_ctx()); // synchronous: work already complete
    return hipSuccess;
}
hipError_t hipEventSynchronize(hipEvent_t /*event*/) {
    ggml_hsa_wait_dispatches(native_ctx());
    return hipSuccess;
}
hipError_t hipEventQuery(hipEvent_t /*event*/) {
    return hipSuccess;
}
hipError_t hipEventElapsedTime(float * ms, hipEvent_t /*a*/, hipEvent_t /*b*/) {
    if (ms) {
        *ms = 0.0f;
    }
    return hipSuccess;
}

} // extern "C"

// ===============================================================================================
// Device management / queries
// ===============================================================================================
extern "C" {

hipError_t hipGetDevice(int * device) {
    *device = 0;
    return hipSuccess;
}
hipError_t hipSetDevice(int /*device*/) {
    return hipSuccess;
}
hipError_t hipGetDeviceCount(int * count) {
    *count = 1;
    return hipSuccess;
}
hipError_t hipSetDeviceFlags(unsigned int /*flags*/) {
    return hipSuccess;
}
hipError_t hipDeviceGetPCIBusId(char * pciBusId, int len, int /*device*/) {
    if (pciBusId && len > 0) {
        std::snprintf(pciBusId, len, "0000:00:00.0");
    }
    return hipSuccess;
}
hipError_t hipDeviceCanAccessPeer(int * canAccess, int /*dev*/, int /*peer*/) {
    *canAccess = 0;
    return hipSuccess;
}
hipError_t hipDeviceEnablePeerAccess(int /*peer*/, unsigned int /*flags*/) {
    return hipSuccess;
}

hipError_t hipDeviceGetAttribute(int * value, hipDeviceAttribute_t attr, int /*device*/) {
    switch (attr) {
        case hipDeviceAttributeWarpSize:
            *value = static_cast<int>(gpu_wavefront_size());
            break;
        case hipDeviceAttributeMaxThreadsPerBlock:
            *value = 1024;
            break;
        case hipDeviceAttributeComputeCapabilityMajor:
            *value = 11;
            break;
        case hipDeviceAttributeComputeCapabilityMinor:
            *value = 5;
            break;
        case hipDeviceAttributeMaxSharedMemoryPerBlock:
            *value = 65536;
            break;
        case hipDeviceAttributeMultiprocessorCount: {
            std::uint32_t cu = 0;
            hsa_agent_get_info(gpu_dev_info().agent,
                               static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
                               &cu);
            *value = cu ? static_cast<int>(cu) : 1;
            break;
        }
        case hipDeviceAttributeIntegrated:
            *value = 1;
            break;
        default:
            *value = 0;
            break;
    }
    return hipSuccess;
}

hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_t * prop, int /*device*/) {
    std::memset(prop, 0, sizeof(*prop));
    std::snprintf(prop->name, sizeof(prop->name), "gfx1151");
    std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "gfx1151");
    prop->warpSize = static_cast<int>(gpu_wavefront_size());
    prop->major = 11;
    prop->minor = 5;
    prop->maxThreadsPerBlock = 1024;
    prop->sharedMemPerBlock = 65536;
    prop->sharedMemPerBlockOptin = 65536;
    prop->integrated = 1;
    std::uint32_t cu = 0;
    hsa_agent_get_info(gpu_dev_info().agent,
                       static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT), &cu);
    prop->multiProcessorCount = cu ? static_cast<int>(cu) : 1;
    std::size_t sz = 0;
    hsa_amd_memory_pool_get_info(gpu_dev_info().data_memory.memory_pool,
                                 HSA_AMD_MEMORY_POOL_INFO_SIZE, &sz);
    prop->totalGlobalMem = sz;
    return hipSuccess;
}

// ---- error handling ----
hipError_t hipGetLastError() {
    return hipSuccess;
}
hipError_t hipPeekAtLastError() {
    return hipSuccess;
}
const char * hipGetErrorString(hipError_t /*err*/) {
    return "hip-abi: no error";
}
const char * hipGetErrorName(hipError_t /*err*/) {
    return "hipSuccess";
}

} // extern "C"

// hipBLAS (hipblasCreate/Destroy/SetStream + GEMM entry points) is implemented in hip_shim_blas.cpp
// on top of a host BLAS over unified memory (used only on the -fa off path).
