#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include <amd_comgr/amd_comgr.h>
#include <assert.h>
#include <climits>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <vector>

#define RET_IF_HSA_ERR(err)                                                                        \
    {                                                                                              \
        if ((err) != HSA_STATUS_SUCCESS) {                                                         \
            std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << __FILE__     \
                      << ". Call returned " << err << std::endl;                                   \
            return (err);                                                                          \
        }                                                                                          \
    }

// ---------------------------------------------------------------------------
// comgr-based kernel argument metadata parsing (prototype)
//
// Instead of hardcoding the COV5 hidden-argument layout, we read the kernel
// argument metadata (.value_kind / .offset / .size) directly from the code
// object using AMD's Code Object Manager (comgr). This is ABI-robust: the
// offsets come from the compiled kernel itself, not a hand-maintained struct.
// ---------------------------------------------------------------------------

struct KernArgInfo {
    std::string value_kind;
    uint32_t    offset;
    uint32_t    size;
};

static bool ComgrGetString(amd_comgr_metadata_node_t node, std::string & out) {
    size_t sz = 0;
    if (amd_comgr_get_metadata_string(node, &sz, nullptr) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }
    std::string s(sz, '\0');
    if (amd_comgr_get_metadata_string(node, &sz, s.data()) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }
    if (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    out = std::move(s);
    return true;
}

static bool ComgrLookupString(amd_comgr_metadata_node_t map, const char * key, std::string & out) {
    amd_comgr_metadata_node_t node;
    if (amd_comgr_metadata_lookup(map, key, &node) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }
    bool ok = ComgrGetString(node, out);
    amd_comgr_destroy_metadata(node);
    return ok;
}

// Parses the argument metadata for kernel descriptor symbol @p symbol (e.g.
// "add_8f32_8f32_8f32.kd") out of the in-memory code object @p blob.
static bool ParseKernelArgs(const std::vector<char> & blob, const std::string & symbol,
                            std::vector<KernArgInfo> & out) {
    amd_comgr_data_t data;
    if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }

    bool result = false;
    bool have_root = false;
    amd_comgr_metadata_node_t root{};
    do {
        if (amd_comgr_set_data(data, blob.size(), blob.data()) != AMD_COMGR_STATUS_SUCCESS) {
            break;
        }
        if (amd_comgr_get_data_metadata(data, &root) != AMD_COMGR_STATUS_SUCCESS) {
            break;
        }
        have_root = true;

        amd_comgr_metadata_node_t kernels;
        if (amd_comgr_metadata_lookup(root, "amdhsa.kernels", &kernels) !=
            AMD_COMGR_STATUS_SUCCESS) {
            break;
        }

        size_t nkernels = 0;
        amd_comgr_get_metadata_list_size(kernels, &nkernels);
        for (size_t i = 0; i < nkernels && !result; ++i) {
            amd_comgr_metadata_node_t kernel;
            if (amd_comgr_index_list_metadata(kernels, i, &kernel) != AMD_COMGR_STATUS_SUCCESS) {
                continue;
            }

            std::string sym;
            if (ComgrLookupString(kernel, ".symbol", sym) && sym == symbol) {
                amd_comgr_metadata_node_t args;
                if (amd_comgr_metadata_lookup(kernel, ".args", &args) == AMD_COMGR_STATUS_SUCCESS) {
                    size_t nargs = 0;
                    amd_comgr_get_metadata_list_size(args, &nargs);
                    for (size_t j = 0; j < nargs; ++j) {
                        amd_comgr_metadata_node_t arg;
                        if (amd_comgr_index_list_metadata(args, j, &arg) !=
                            AMD_COMGR_STATUS_SUCCESS) {
                            continue;
                        }
                        std::string kind, off, size;
                        ComgrLookupString(arg, ".value_kind", kind);
                        ComgrLookupString(arg, ".offset", off);
                        ComgrLookupString(arg, ".size", size);
                        KernArgInfo info;
                        info.value_kind = kind;
                        info.offset = off.empty() ? 0u : static_cast<uint32_t>(std::stoul(off));
                        info.size = size.empty() ? 0u : static_cast<uint32_t>(std::stoul(size));
                        out.push_back(info);
                        amd_comgr_destroy_metadata(arg);
                    }
                    amd_comgr_destroy_metadata(args);
                    result = !out.empty();
                }
            }
            amd_comgr_destroy_metadata(kernel);
        }
        amd_comgr_destroy_metadata(kernels);
    } while (false);

    if (have_root) {
        amd_comgr_destroy_metadata(root);
    }
    amd_comgr_release_data(data);
    return result;
}

// Reads a file fully into a byte buffer.
static bool ReadFileBytes(const std::string & path, std::vector<char> & out) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is) {
        return false;
    }
    const std::streamsize size = is.tellg();
    is.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    return static_cast<bool>(is.read(out.data(), size));
}

static uint32_t kAddN = 8;  // Default value, can be overridden by command-line argument
static const uint32_t kWorkGroupSize = 64;  // Workgroup size

typedef struct AddStruct {

    // Kernel argument buffers
    float * h_a;
    float * h_b;
    float * d_a;
    float * d_b;
    float * output;
    uint64_t N;

    // Kernel parameters
    uint32_t work_group_size;
    uint32_t work_grid_size;

    // Keneral argument buffers and addresses
    void * kern_arg_buffer; // Begin of allocated memory
    //  this pointer to be deallocated
    void * kern_arg_address; // Properly aligned address to be used in aql
    // packet (don't use for deallocation)

    // Kernel code
    std::string kernel_file_name;
    std::string kernel_name;
    uint32_t kernarg_size;
    uint32_t kernarg_align;

    // HSA/RocR objects needed for this application
    hsa_agent_t gpu_dev;
    hsa_agent_t cpu_dev;
    hsa_signal_t signal;
    hsa_queue_t * queue;
    hsa_amd_memory_pool_t cpu_pool;
    hsa_amd_memory_pool_t gpu_pool;
    hsa_amd_memory_pool_t kern_arg_pool;

    // Other items we need to populate AQL packet
    uint64_t kernel_object;
    uint32_t group_segment_size;   ///< Kernel group seg size
    uint32_t private_segment_size; ///< Kernel private seg size

} AddStruct;

void InitializeAdd(AddStruct * add, uint32_t vector_size) {
    // Build kernel file name and kernel name based on vector size
    // e.g., for vector_size=8: "add_8f32_8f32_8f32.hsaco" and "add_8f32_8f32_8f32.kd"
    std::string size_str = std::to_string(vector_size);
    add->kernel_file_name = "add_" + size_str + "f32_" + size_str + "f32_" + size_str + "f32.hsaco";
    add->kernel_name = "add_" + size_str + "f32_" + size_str + "f32_" + size_str + "f32.kd";
    add->N = vector_size;
    add->work_group_size = kWorkGroupSize;
    // Calculate grid size to cover all N elements
    // Grid size must be a multiple of work group size
    add->work_grid_size = ((vector_size + kWorkGroupSize - 1) / kWorkGroupSize) * kWorkGroupSize;
}

// This function is called by the call-back functions used to find an agent of
// the specified hsa_device_type_t. Note that it cannot be called directly from
// hsa_iterate_agents() as it does not match the prototype of the call-back
// function. It must be wrapped by a function with the correct prototype.
//
// Return values:
//  HSA_STATUS_INFO_BREAK -- "agent" is of the specified type (dev_type)
//  HSA_STATUS_SUCCESS -- "agent" is not of the specified type
//  Other -- Some error occurred
static hsa_status_t FindAgent(hsa_agent_t agent, void * data, hsa_device_type_t dev_type) {
    if (data == nullptr) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    // See if the provided agent matches the input type (dev_type)
    hsa_device_type_t hsa_device_type;
    hsa_status_t hsa_error_code =
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
    RET_IF_HSA_ERR(hsa_error_code);

    if (hsa_device_type == dev_type) {
        *(reinterpret_cast<hsa_agent_t *>(data)) = agent;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

// This is the call-back function used to find a GPU type agent. Note that the
// prototype of this function is dictated by the HSA specification
hsa_status_t FindGPUDevice(hsa_agent_t agent, void * data) {
    return FindAgent(agent, data, HSA_DEVICE_TYPE_GPU);
}

// This is the call-back function used to find a CPU type agent. Note that the
// prototype of this function is dictated by the HSA specification
hsa_status_t FindCPUDevice(hsa_agent_t agent, void * data) {
    return FindAgent(agent, data, HSA_DEVICE_TYPE_CPU);
}

// Find the CPU and GPU agents we need to run this sample, and save them in the
// AddStruct structure for later use.
hsa_status_t FindDevices(AddStruct * add) {
    hsa_status_t err;

    // Note that hsa_iterate_agents iterate through all known agents until
    // HSA_STATUS_SUCCESS is not returned. The call-backs are implemented such
    // that HSA_STATUS_INFO_BREAK means we found an agent of the specified type.
    // This value is returned by hsa_iterate_agents.
    add->gpu_dev.handle = 0;
    err = hsa_iterate_agents(FindGPUDevice, &add->gpu_dev);

    if (err != HSA_STATUS_INFO_BREAK) {
        return HSA_STATUS_ERROR;
    }

    add->cpu_dev.handle = 0;
    err = hsa_iterate_agents(FindCPUDevice, &add->cpu_dev);

    if (err != HSA_STATUS_INFO_BREAK) {
        return HSA_STATUS_ERROR;
    }

    if (0 == add->gpu_dev.handle) {
        std::cout << "GPU Device is not Created properly!" << std::endl;
        RET_IF_HSA_ERR(HSA_STATUS_ERROR);
    }

    if (0 == add->cpu_dev.handle) {
        std::cout << "CPU Device is not Created properly!" << std::endl;
        RET_IF_HSA_ERR(HSA_STATUS_ERROR);
    }

    return HSA_STATUS_SUCCESS;
}

// This function checks to see if the provided
// pool has the HSA_AMD_SEGMENT_GLOBAL property. If the kern_arg flag is true,
// the function adds an additional requirement that the pool have the
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT property. If kern_arg is false,
// pools must NOT have this property.
// Upon finding a pool that meets these conditions, HSA_STATUS_INFO_BREAK is
// returned. HSA_STATUS_SUCCESS is returned if no errors were encountered, but
// no pool was found meeting the requirements. If an error is encountered, we
// return that error.

// Note that this function does not match the required prototype for the
// hsa_amd_agent_iterate_memory_pools call back function, and therefore must be
// wrapped by a function with the correct prototype.
static hsa_status_t FindGlobalPool(hsa_amd_memory_pool_t pool, void * data, bool kern_arg) {
    hsa_status_t err;
    hsa_amd_segment_t segment;
    uint32_t flag;

    if (nullptr == data) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    RET_IF_HSA_ERR(err);

    if (HSA_AMD_SEGMENT_GLOBAL != segment) {
        return HSA_STATUS_SUCCESS;
    }

    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
    RET_IF_HSA_ERR(err);

    uint32_t karg_st = flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;

    if ((karg_st == 0 && kern_arg) || (karg_st != 0 && !kern_arg)) {
        return HSA_STATUS_SUCCESS;
    }

    *(reinterpret_cast<hsa_amd_memory_pool_t *>(data)) = pool;
    return HSA_STATUS_INFO_BREAK;
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that is NOT
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t FindStandardPool(hsa_amd_memory_pool_t pool, void * data) {
    return FindGlobalPool(pool, data, false);
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that IS
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t FindKernArgPool(hsa_amd_memory_pool_t pool, void * data) {
    return FindGlobalPool(pool, data, true);
}

// Find memory pools that we will need to allocate from for this sample
// application. We will need memory associated with the host CPU, the GPU
// executing the kernels, and for kernel arguments. This function will
// save the found pools to the AddStruct structure for use elsewhere
// in this program.
hsa_status_t FindPools(AddStruct * add) {
    hsa_status_t err;

    err = hsa_amd_agent_iterate_memory_pools(add->cpu_dev, FindStandardPool, &add->cpu_pool);

    if (err != HSA_STATUS_INFO_BREAK) {
        return HSA_STATUS_ERROR;
    }

    err = hsa_amd_agent_iterate_memory_pools(add->gpu_dev, FindStandardPool, &add->gpu_pool);

    if (err != HSA_STATUS_INFO_BREAK) {
        return HSA_STATUS_ERROR;
    }

    err = hsa_amd_agent_iterate_memory_pools(add->cpu_dev, FindKernArgPool, &add->kern_arg_pool);

    if (err != HSA_STATUS_INFO_BREAK) {
        return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
}

// Once the needed memory pools have been found and the AddStruct structure
// has been updated with these handles, this function is then used to allocate
// memory from those pools.
// Devices with which a pool is associated already have access to the pool.
// However, other devices may also need to read or write to that memory. Below,
// we see how we can grant access to other devices to address this issue.
hsa_status_t AllocateAndInitBuffers(AddStruct * add) {
    hsa_status_t err;
    size_t out_length = add->N * sizeof(float);
    size_t in_length = add->N * sizeof(float);

    // In all of these examples, we want both the cpu and gpu to have access to
    // the buffer in question. We use the array of agents below in the susequent
    // calls to hsa_amd_agents_allow_access() for this purpose.
    hsa_agent_t ag_list[2] = {add->gpu_dev, add->cpu_dev};

    err = hsa_amd_memory_pool_allocate(add->cpu_pool, in_length, 0,
                                       reinterpret_cast<void **>(&add->h_a));
    RET_IF_HSA_ERR(err);
    err = hsa_amd_agents_allow_access(2, ag_list, NULL, add->h_a);
    RET_IF_HSA_ERR(err);
    (void)memset(add->h_a, 0, in_length);

    err = hsa_amd_memory_pool_allocate(add->cpu_pool, in_length, 0,
                                       reinterpret_cast<void **>(&add->h_b));
    RET_IF_HSA_ERR(err);
    err = hsa_amd_agents_allow_access(2, ag_list, NULL, add->h_b);
    RET_IF_HSA_ERR(err);
    (void)memset(add->h_b, 0, in_length);

    err = hsa_amd_memory_pool_allocate(add->cpu_pool, in_length, 0,
                                       reinterpret_cast<void **>(&add->d_a));
    RET_IF_HSA_ERR(err);
    err = hsa_amd_agents_allow_access(2, ag_list, NULL, add->d_a);
    RET_IF_HSA_ERR(err);
    (void)memset(add->d_a, 0, in_length);

    err = hsa_amd_memory_pool_allocate(add->cpu_pool, in_length, 0,
                                       reinterpret_cast<void **>(&add->d_b));
    RET_IF_HSA_ERR(err);
    err = hsa_amd_agents_allow_access(2, ag_list, NULL, add->d_b);
    RET_IF_HSA_ERR(err);
    (void)memset(add->d_b, 0, in_length);

    err = hsa_amd_memory_pool_allocate(add->cpu_pool, out_length, 0,
                                       reinterpret_cast<void **>(&add->output));
    RET_IF_HSA_ERR(err);
    err = hsa_amd_agents_allow_access(2, ag_list, NULL, add->output);
    RET_IF_HSA_ERR(err);
    (void)memset(add->output, 0, in_length);

    // Binary-search application specific code...
    // Initialize input buffer with random values in an increasing order
    const uint64_t N = add->N;
    // add->d_a = new float[N];
    // add->d_b = new float[N];
    for (uint64_t i = 0; i < N; ++i) {
        add->h_a[i] = i * 1.0f;
        add->d_a[i] = i * 1.0f;
        add->h_b[i] = i * 2.0f;
        add->d_b[i] = i * 2.0f;
    }

    return err;
}

// The code in this function illustrates how to load a kernel from
// pre-compiled code. The goal is to get a handle that can be later
// used in an AQL packet and also to extract information about kernel
// that we will need. All of the information hand kernel handle will
// be saved to the AddStruct structure. It will be used when we
// populate the AQL packet.
hsa_status_t LoadKernelFromObjFile(AddStruct * add) {
    hsa_status_t err;
    hsa_code_object_reader_t code_obj_rdr = {0};
    hsa_executable_t executable = {0};

    hsa_file_t file_handle = open(add->kernel_file_name.c_str(), O_RDONLY);

    if (file_handle == -1) {
        char agent_name[64];
        err = hsa_agent_get_info(add->gpu_dev, HSA_AGENT_INFO_NAME, agent_name);
        RET_IF_HSA_ERR(err);
        std::string fileName = std::string("./") + agent_name + "/" + add->kernel_file_name;
        file_handle = open(fileName.c_str(), O_RDONLY);  // Don't redeclare, assign to existing variable
    }

    if (file_handle == -1) {
        std::cout << "failed to open " << add->kernel_file_name.c_str() << " at line " << __LINE__
                  << ", errno: " << errno << std::endl;
        return HSA_STATUS_ERROR;
    }

    err = hsa_code_object_reader_create_from_file(file_handle, &code_obj_rdr);
    close(file_handle);
    RET_IF_HSA_ERR(err);

    err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
                                    &executable);
    RET_IF_HSA_ERR(err);

    err = hsa_executable_load_agent_code_object(executable, add->gpu_dev, code_obj_rdr, NULL, NULL);
    RET_IF_HSA_ERR(err);

    err = hsa_executable_freeze(executable, NULL);
    RET_IF_HSA_ERR(err);

    hsa_executable_symbol_t kern_sym;
    err = hsa_executable_get_symbol(executable, NULL, add->kernel_name.c_str(), add->gpu_dev, 0,
                                    &kern_sym);
    RET_IF_HSA_ERR(err);

    err = hsa_executable_symbol_get_info(kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                         &add->kernel_object);
    RET_IF_HSA_ERR(err);

    err = hsa_executable_symbol_get_info(kern_sym,
                                         HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                         &add->private_segment_size);
    RET_IF_HSA_ERR(err);

    err = hsa_executable_symbol_get_info(
        kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &add->group_segment_size);
    RET_IF_HSA_ERR(err);

    // Remaining queries not supported on code object v3.
    err = hsa_executable_symbol_get_info(
        kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &add->kernarg_size);
    RET_IF_HSA_ERR(err);

    err = hsa_executable_symbol_get_info(
        kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &add->kernarg_align);
    RET_IF_HSA_ERR(err);
    assert(add->kernarg_align >= 16 && "Reported kernarg size is too small.");
    add->kernarg_align = (add->kernarg_align == 0) ? 16 : add->kernarg_align;

    return err;
}

// AlignDown and AlignUp are 2 utility functions we use to find an aligned
// boundary either below or above a given value (address). The function will
// return a value that has the specified alignment.
static intptr_t AlignDown(intptr_t value, size_t alignment) {
    assert(alignment != 0 && "Zero alignment");
    return (intptr_t)(value & ~(alignment - 1));
}
static void * AlignUp(void * value, size_t alignment) {
    return reinterpret_cast<void *>(
        AlignDown((uintptr_t)(reinterpret_cast<uintptr_t>(value) + alignment - 1), alignment));
}

// This function allocates memory from the kern_arg pool we already found, and
// then sets the argument values needed by the kernel code.
hsa_status_t
AllocAndSetKernArgs(AddStruct * bs, void * args, size_t arg_size, void ** aql_buf_ptr) {
    void * kern_arg_buf = nullptr;
    hsa_status_t err;
    size_t buf_size;
    size_t req_align;

    // The kernel code must be written to memory at the correct alignment. We
    // already queried the executable to get the correct alignment, which is
    // stored in bs->kernarg_align. In case the memory returned from
    // hsa_amd_memory_pool is not of the correct alignment, we request a little
    // more than what we need in case we need to adjust.
    req_align = bs->kernarg_align;
    // Allocate enough extra space for alignment adjustments if ncessary
    buf_size = (arg_size >= bs->kernarg_size ? arg_size : bs->kernarg_size) + (req_align << 1);

    err = hsa_amd_memory_pool_allocate(bs->kern_arg_pool, buf_size, 0,
                                       reinterpret_cast<void **>(&kern_arg_buf));
    RET_IF_HSA_ERR(err);

    // Address of the allocated buffer
    bs->kern_arg_buffer = kern_arg_buf;

    // Addr. of kern arg start.
    bs->kern_arg_address = AlignUp(kern_arg_buf, req_align);

    std::cout << "arg_size = " << arg_size << ", bs->kernarg_size = " << bs->kernarg_size
              << std::endl;
    // assert(arg_size >= bs->kernarg_size);
    // assert(((uintptr_t)bs->kern_arg_address + arg_size) <
    //        ((uintptr_t)bs->kern_arg_buffer + buf_size));

    (void)memcpy(bs->kern_arg_address, args, arg_size);
    RET_IF_HSA_ERR(err);

    // Make sure both the CPU and GPU can access the kernel arguments
    hsa_agent_t ag_list[2] = {bs->gpu_dev, bs->cpu_dev};
    err = hsa_amd_agents_allow_access(2, ag_list, NULL, bs->kern_arg_buffer);
    RET_IF_HSA_ERR(err);

    // Save this info in our BinarySearch structure for later.
    *aql_buf_ptr = bs->kern_arg_address;

    return HSA_STATUS_SUCCESS;
}

void PopulateAQLPacket(AddStruct const * bs, hsa_kernel_dispatch_packet_t * aql) {

    uint16_t const ndim = 1;

    aql->header = 0; // Dummy val. for now. Set this right before doorbell ring
    aql->setup = ndim << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    aql->workgroup_size_x = bs->work_group_size;
    aql->workgroup_size_y = 1;
    aql->workgroup_size_z = 1;
    aql->grid_size_x = bs->work_grid_size;
    aql->grid_size_y = 1;
    aql->grid_size_z = 1;
    aql->private_segment_size = bs->private_segment_size;
    aql->group_segment_size = bs->group_segment_size;
    aql->kernel_object = bs->kernel_object;
    aql->kernarg_address = bs->kern_arg_address;
    aql->completion_signal = bs->signal;

    return;
}

// This function shows how to do an asynchronous copy. We have to create a
// signal and use the signal to notify us when the copy has completed.
hsa_status_t
AgentMemcpy(void * dst, const void * src, size_t size, hsa_agent_t dst_ag, hsa_agent_t src_ag) {
    hsa_signal_t s;
    hsa_status_t err;

    err = hsa_signal_create(1, 0, NULL, &s);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_async_copy(dst, dst_ag, src, src_ag, size, 0, NULL, s);
    RET_IF_HSA_ERR(err);

    if (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                  HSA_WAIT_STATE_BLOCKED) != 0) {
        err = HSA_STATUS_ERROR;
        std::cout << "Async copy signal error" << std::endl;

        RET_IF_HSA_ERR(err);
    }

    err = hsa_signal_destroy(s);

    RET_IF_HSA_ERR(err);

    return err;
}

/*
 * Write everything in the provided AQL packet to the queue except the first 32
 * bits which include the header and setup fields. That should be done
 * last.
 * Note: The caller is responsible for managing the queue write index.
 */
void WriteAQLToQueue(hsa_kernel_dispatch_packet_t const * in_aql, hsa_queue_t * q, uint64_t que_idx) {
    void * queue_base = q->base_address;
    const uint32_t queue_mask = q->size - 1;

    hsa_kernel_dispatch_packet_t * queue_aql_packet;

    queue_aql_packet =
        &(reinterpret_cast<hsa_kernel_dispatch_packet_t *>(queue_base))[que_idx & queue_mask];

    queue_aql_packet->workgroup_size_x = in_aql->workgroup_size_x;
    queue_aql_packet->workgroup_size_y = in_aql->workgroup_size_y;
    queue_aql_packet->workgroup_size_z = in_aql->workgroup_size_z;
    queue_aql_packet->grid_size_x = in_aql->grid_size_x;
    queue_aql_packet->grid_size_y = in_aql->grid_size_y;
    queue_aql_packet->grid_size_z = in_aql->grid_size_z;
    queue_aql_packet->private_segment_size = in_aql->private_segment_size;
    queue_aql_packet->group_segment_size = in_aql->group_segment_size;
    queue_aql_packet->kernel_object = in_aql->kernel_object;
    queue_aql_packet->kernarg_address = in_aql->kernarg_address;
    queue_aql_packet->completion_signal = in_aql->completion_signal;
}

// This wrapper atomically writes the provided header and setup to the
// provided AQL packet. The provided AQL packet address should be in the
// queue memory space.
inline void AtomicSetPacketHeader(uint16_t header,
                                  uint16_t setup,
                                  hsa_kernel_dispatch_packet_t * queue_packet) {
    __atomic_store_n(reinterpret_cast<uint32_t *>(queue_packet), header | (setup << 16),
                     __ATOMIC_RELEASE);
}

// Once all the required data for kernel execution is collected (in this
// application it is stored in the BinarySearch structure) we can put it in
// an AQL packet and ring the queue door bell to tell the command processor to
// execute it.
hsa_status_t Run(AddStruct * bs) {
    hsa_status_t err;

    std::cout << "Executing kernel " << bs->kernel_name << std::endl;

    // Metadata-driven kernarg layout: read the argument offsets from the code
    // object itself (via comgr) instead of hardcoding the COV5 struct.
    std::vector<char> blob;
    if (!ReadFileBytes(bs->kernel_file_name, blob)) {
        std::cout << "failed to read code object " << bs->kernel_file_name << std::endl;
        return HSA_STATUS_ERROR;
    }
    std::vector<KernArgInfo> arg_info;
    if (!ParseKernelArgs(blob, bs->kernel_name, arg_info)) {
        std::cout << "failed to parse arg metadata for " << bs->kernel_name << std::endl;
        return HSA_STATUS_ERROR;
    }

    const uint32_t num_blocks_x = (bs->N + bs->work_group_size - 1) / bs->work_group_size;

    // Build the kernarg segment from the metadata. The whole segment is zeroed
    // first so any hidden field we do not explicitly set (e.g. global_offset) is
    // well-defined.
    std::vector<uint8_t> kernarg(bs->kernarg_size, 0);
    auto put = [&](uint32_t off, const void * src, uint32_t sz) {
        if (static_cast<size_t>(off) + sz <= kernarg.size()) {
            memcpy(kernarg.data() + off, src, sz);
        }
    };

    int global_buffer_idx = 0;
    void * const buffers[3] = {bs->d_a, bs->d_b, bs->output};
    for (const auto & a : arg_info) {
        if (a.value_kind == "global_buffer") {
            uint64_t v = reinterpret_cast<uint64_t>(global_buffer_idx < 3 ? buffers[global_buffer_idx]
                                                                          : nullptr);
            put(a.offset, &v, a.size);
            ++global_buffer_idx;
        } else if (a.value_kind == "by_value") {
            uint64_t v = bs->N; // the only by-value arg is N
            put(a.offset, &v, a.size);
        } else if (a.value_kind == "hidden_block_count_x") {
            uint32_t v = num_blocks_x;
            put(a.offset, &v, a.size);
        } else if (a.value_kind == "hidden_block_count_y" ||
                   a.value_kind == "hidden_block_count_z") {
            uint32_t v = 1;
            put(a.offset, &v, a.size);
        } else if (a.value_kind == "hidden_group_size_x") {
            uint16_t v = static_cast<uint16_t>(bs->work_group_size);
            put(a.offset, &v, a.size);
        } else if (a.value_kind == "hidden_group_size_y" ||
                   a.value_kind == "hidden_group_size_z") {
            uint16_t v = 1;
            put(a.offset, &v, a.size);
        } else if (a.value_kind == "hidden_remainder_x") {
            uint16_t v = static_cast<uint16_t>(bs->N % bs->work_group_size);
            put(a.offset, &v, a.size);
        } else if (a.value_kind == "hidden_grid_dims") {
            uint16_t v = 1; // 1D grid
            put(a.offset, &v, a.size);
        }
        // all other hidden args (remainder_y/z, global_offset_*, etc.) stay zero
    }

    std::cout << "kernarg_size = " << bs->kernarg_size << ", parsed " << arg_info.size()
              << " args" << std::endl;
    err = AllocAndSetKernArgs(bs, kernarg.data(), kernarg.size(), &bs->kern_arg_address);
    RET_IF_HSA_ERR(err);

    // Populate an AQL packet with the info we've gathered
    hsa_kernel_dispatch_packet_t aql;
    PopulateAQLPacket(bs, &aql);

    // std::cout << "Dispatch info: N=" << bs->N 
    //           << ", work_group_size=" << bs->work_group_size 
    //           << ", work_grid_size=" << bs->work_grid_size 
    //           << ", num_workgroups=" << (bs->work_grid_size / bs->work_group_size) << std::endl;

    // // Copy kernel parameter from system memory to local memory
    // size_t in_length = bs->N * sizeof(float);
    // err = AgentMemcpy(reinterpret_cast<float *>(bs->d_a), reinterpret_cast<float *>(bs->h_a),
    //                   in_length, bs->gpu_dev, bs->cpu_dev);

    // RET_IF_HSA_ERR(err);

    // // Copy kernel parameter from system memory to local memory
    // err = AgentMemcpy(reinterpret_cast<float *>(bs->d_b), reinterpret_cast<float *>(bs->h_b),
    //                   in_length, bs->gpu_dev, bs->cpu_dev);
    // RET_IF_HSA_ERR(err);

    for (uint64_t i = 0; i < bs->N; ++i) {
        bs->output[i] = 0.0f;
    }

    // Dispatch kernel with global work size, work group size with ONE dimesion
    // and wait for kernel to complete

    // Compute the write index of queue and copy Aql packet into it
    uint64_t que_idx = hsa_queue_load_write_index_relaxed(bs->queue);

    const uint32_t mask = bs->queue->size - 1;

    // This function simply copies the data we've collected so far into our
    // local AQL packet, except the the setup and header fields.
    WriteAQLToQueue(&aql, bs->queue, que_idx);

    // Insert a release fence to ensure all memory operations before dispatch
    __atomic_thread_fence(__ATOMIC_ACQ_REL);

    uint32_t aql_header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    // Set the packet's type, acquire and release fences. This should be done
    // atomically after all the other fields have been set, using release
    // memory ordering to ensure all the fields are set when the door bell
    // signal is activated.
    void * q_base = bs->queue->base_address;

    AtomicSetPacketHeader(
        aql_header, aql.setup,
        &(reinterpret_cast<hsa_kernel_dispatch_packet_t *>(q_base))[que_idx & mask]);

    // Increment the write index and ring the doorbell to dispatch kernel.
    hsa_queue_store_write_index_relaxed(bs->queue, (que_idx + 1));
    hsa_signal_store_relaxed(bs->queue->doorbell_signal, que_idx);

    // Wait on the dispatch signal until the kernel is finished.
    // Modify the wait condition to HSA_WAIT_STATE_ACTIVE (instead of
    // HSA_WAIT_STATE_BLOCKED) if polling is needed instead of blocking, as we
    // have below.
    // The call below will block until the condition is met. Below we have said
    // the condition is that the signal value (initiailzed to 1) associated with
    // the queue is less than 1. When the kernel associated with the queued AQL
    // packet has completed execution, the signal value is automatically
    // decremented by the packet processor.
    hsa_signal_value_t value = hsa_signal_wait_scacquire(bs->signal, HSA_SIGNAL_CONDITION_LT, 1,
                                                         UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // value should be 0, or we timed-out
    if (value) {
        std::cout << "Timed out waiting for kernel to complete?" << std::endl;
        RET_IF_HSA_ERR(HSA_STATUS_ERROR);
    }

    // Reset the signal to its initial value for the next iteration
    hsa_signal_store_screlease(bs->signal, 1);

    std::cout << "Verifying kernel output..." << std::endl;
    for (uint64_t i = 0; i < bs->N; ++i) {
        float expected = 3.0f * i;
        if (bs->output[i] != expected) {
            std::cout << "[" << i << "]: FAIL - expected " << expected << ", got " << bs->output[i]
                      << std::endl;
            return HSA_STATUS_ERROR;
        }
    }
    std::cout << "All " << bs->N << " elements verified correctly (output[i] == 3*i)" << std::endl;

    return HSA_STATUS_SUCCESS;
}

// Release all the RocR resources we have acquired in this application.
hsa_status_t CleanUp(AddStruct * add) {
    hsa_status_t err;

    err = hsa_amd_memory_pool_free(add->h_a);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_free(add->h_b);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_free(add->d_a);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_free(add->d_b);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_free(add->output);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_free(add->kern_arg_buffer);
    RET_IF_HSA_ERR(err);

    err = hsa_queue_destroy(add->queue);
    RET_IF_HSA_ERR(err);

    err = hsa_signal_destroy(add->signal);
    RET_IF_HSA_ERR(err);

    err = hsa_shut_down();
    RET_IF_HSA_ERR(err);

    return HSA_STATUS_SUCCESS;
}

int main(int argc, char * argv[]) {

    AddStruct add;
    hsa_status_t err;

    // Parse command-line argument for vector size
    uint32_t vector_size = kAddN;  // Default value
    if (argc > 1) {
        vector_size = static_cast<uint32_t>(std::stoul(argv[1]));
        if (vector_size == 0) {
            std::cerr << "Error: vector size must be greater than 0" << std::endl;
            return 1;
        }
    }
    kAddN = vector_size;  // Update global for any other uses

    std::cout << "Using vector size: " << vector_size << std::endl;

    InitializeAdd(&add, vector_size);

    // hsa_init() initializes internal data structures and causes devices
    // (agents), memory pools and other resources to be discovered.
    err = hsa_init();
    RET_IF_HSA_ERR(err);

    // Find the agents needed for the sample
    err = FindDevices(&add);
    RET_IF_HSA_ERR(err);

    // Create the completion signal used when dispatching a packet
    err = hsa_signal_create(1, 0, NULL, &add.signal);
    RET_IF_HSA_ERR(err);

    // Create a queue to submit our binary search AQL packets
    err = hsa_queue_create(add.gpu_dev, 128, HSA_QUEUE_TYPE_MULTI, NULL, NULL, UINT32_MAX,
                           UINT32_MAX, &add.queue);
    RET_IF_HSA_ERR(err);

    // Find the HSA memory pools we need to run this sample
    err = FindPools(&add);
    RET_IF_HSA_ERR(err);

    // Allocate memory from the correct memory pool, and initialize them as
    // neeeded for the algorihm.
    err = AllocateAndInitBuffers(&add);
    RET_IF_HSA_ERR(err);

    // Create a kernel object from the pre-compiled kernel, and read some
    // attributes associated with the kernel that we will need.
    err = LoadKernelFromObjFile(&add);
    RET_IF_HSA_ERR(err);

    // Fill in the AQL packet, assign the kernel arguments, enqueue the packet,
    // "ring" the doorbell, and wait for completion.
    err = Run(&add);
    RET_IF_HSA_ERR(err);

    // Release all the RocR resources we've acquired and shutdown HSA.
    err = CleanUp(&add);

    return 0;
}
