// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "hsaco-metadata.hpp"

#include <amd_comgr/amd_comgr.h>

namespace {

/**
 * @brief Reads the string value of a comgr metadata node.
 */
bool comgr_get_string(amd_comgr_metadata_node_t node, std::string & out) {
    std::size_t sz = 0;
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

/**
 * @brief Looks up a string-valued key in a comgr metadata map.
 */
bool comgr_lookup_string(amd_comgr_metadata_node_t map, const char * key, std::string & out) {
    amd_comgr_metadata_node_t node;
    if (amd_comgr_metadata_lookup(map, key, &node) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }
    bool ok = comgr_get_string(node, out);
    amd_comgr_destroy_metadata(node);
    return ok;
}

} // namespace

bool ggml_hsaco_parse_kernels(const void * code_object, std::size_t size,
                              std::unordered_map<std::string, ggml_hsaco_kernel> & out) {
    amd_comgr_data_t data;
    if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }

    bool have_root = false;
    amd_comgr_metadata_node_t root{};
    do {
        if (amd_comgr_set_data(data, size, static_cast<const char *>(code_object)) !=
            AMD_COMGR_STATUS_SUCCESS) {
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

        std::size_t nkernels = 0;
        amd_comgr_get_metadata_list_size(kernels, &nkernels);
        for (std::size_t i = 0; i < nkernels; ++i) {
            amd_comgr_metadata_node_t kernel;
            if (amd_comgr_index_list_metadata(kernels, i, &kernel) != AMD_COMGR_STATUS_SUCCESS) {
                continue;
            }

            ggml_hsaco_kernel info;
            if (comgr_lookup_string(kernel, ".symbol", info.symbol)) {
                amd_comgr_metadata_node_t args;
                if (amd_comgr_metadata_lookup(kernel, ".args", &args) == AMD_COMGR_STATUS_SUCCESS) {
                    std::size_t nargs = 0;
                    amd_comgr_get_metadata_list_size(args, &nargs);
                    for (std::size_t j = 0; j < nargs; ++j) {
                        amd_comgr_metadata_node_t arg;
                        if (amd_comgr_index_list_metadata(args, j, &arg) !=
                            AMD_COMGR_STATUS_SUCCESS) {
                            continue;
                        }
                        std::string name, kind, off, sz;
                        comgr_lookup_string(arg, ".name", name); // explicit args only
                        comgr_lookup_string(arg, ".value_kind", kind);
                        comgr_lookup_string(arg, ".offset", off);
                        comgr_lookup_string(arg, ".size", sz);
                        ggml_hsaco_arg a;
                        a.name = std::move(name);
                        a.value_kind = std::move(kind);
                        a.offset = off.empty() ? 0u : static_cast<std::uint32_t>(std::stoul(off));
                        a.size = sz.empty() ? 0u : static_cast<std::uint32_t>(std::stoul(sz));
                        info.args.push_back(std::move(a));
                        amd_comgr_destroy_metadata(arg);
                    }
                    amd_comgr_destroy_metadata(args);
                }

                // Key = mangled name (symbol with the ".kd" descriptor suffix stripped).
                std::string key = info.symbol;
                if (key.size() > 3 && key.compare(key.size() - 3, 3, ".kd") == 0) {
                    key.resize(key.size() - 3);
                }
                if (!key.empty()) {
                    out.emplace(std::move(key), std::move(info));
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
    return !out.empty();
}
