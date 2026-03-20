#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <stdint.h>

namespace edgedeploy {
namespace inference {

// Qualcomm QNN Execution Provider (EP) Configuration Options
// References the Snapdragon 8 Elite (SM8750) Hexagon HTP NPU & Oryon CPU Core layout
enum class NpuPrecision {
    FP32,
    FP16,
    INT8_DYNAMIC,
    INT8_STATIC
};

enum class PerformanceProfile {
    LOW_POWER,
    BALANCED,
    HIGH_PERFORMANCE,
    BURST,
    SUSTAINED_HIGH_PERFORMANCE
};

struct QnnEPConfig {
    std::string backend_path = "libQnnHtp.so"; // Default to Hexagon HTP Backend
    PerformanceProfile perf_profile = PerformanceProfile::BURST;
    NpuPrecision precision = NpuPrecision::INT8_DYNAMIC;
    
    // HTP specific settings
    bool use_htp_fp16_precision = true;
    bool enable_vertex_vector_extensions = true;
    int htp_npu_device_id = 0;
    
    // Context compilation caching for fast start-up times
    bool enable_context_caching = true;
    std::string context_cache_dir = "/data/local/tmp/qnn_context_cache";
    std::string model_identifier = "indic_conformer_120m";
    
    // Qualcomm QNN HTP power configurations
    uint32_t dsp_voltage_corner = 2; // High-performance corner (Hvx/HTP voltage)
    uint32_t rpc_latency_us = 0;     // Minimum RPC latency for ultra-low latency execution
};

// Snapdragon Oryon CPU Cluster Configurations
// Snapdragon 8 Elite (SM8750): 2x Prime Cores (up to 4.32 GHz), 6x Performance Cores (up to 3.53 GHz)
// Snapdragon 8 Gen 2 (SM8550): 1x Prime Core, 4x Performance Cores, 3x Efficiency Cores
struct CpuAffinityConfig {
    std::vector<int> target_cores; // Vector of CPU core IDs to bind to
    bool exclude_efficiency_cores = true;
    bool prioritize_prime_cores = true;
    
    // Thread pool size for ONNX Runtime intra-op execution
    int intra_op_num_threads = 4;
};

// Dynamic helper to construct key-value options for ONNX Runtime SessionOptions
inline std::unordered_map<std::string, std::string> GetQnnEpOptions(const QnnEPConfig& config) {
    std::unordered_map<std::string, std::string> options;
    options["backend_path"] = config.backend_path;
    
    // Map performance profile to ORT QNN EP keys
    std::string profile_str = "BURST";
    switch (config.perf_profile) {
        case PerformanceProfile::LOW_POWER: profile_str = "LOW_POWER"; break;
        case PerformanceProfile::BALANCED: profile_str = "BALANCED"; break;
        case PerformanceProfile::HIGH_PERFORMANCE: profile_str = "HIGH_PERFORMANCE"; break;
        case PerformanceProfile::BURST: profile_str = "BURST"; break;
        case PerformanceProfile::SUSTAINED_HIGH_PERFORMANCE: profile_str = "SUSTAINED_HIGH_PERFORMANCE"; break;
    }
    options["performance_mode"] = profile_str;
    
    // Precision mapping
    options["enable_htp_fp16"] = config.use_htp_fp16_precision ? "1" : "0";
    options["enable_htp_vertex_vector"] = config.enable_vertex_vector_extensions ? "1" : "0";
    
    // Context caching
    if (config.enable_context_caching) {
        options["context_enable_caching"] = "1";
        options["context_cache_path"] = config.context_cache_dir;
        options["context_cache_prefix"] = config.model_identifier;
    } else {
        options["context_enable_caching"] = "0";
    }
    
    // HTP Power optimization params
    options["htp_performance_mode"] = profile_str;
    options["rpc_control_latency"] = std::to_string(config.rpc_latency_us);
    options["htp_voltage_corner"] = std::to_string(config.dsp_voltage_corner);
    
    return options;
}

} // namespace inference
} // namespace edgedeploy
