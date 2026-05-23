#pragma once

#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "qnn_configs.h"

namespace edgedeploy {
namespace inference {

class ASREngine {
public:
    ASREngine();
    ~ASREngine();

    // Initializes the ONNX Runtime environment, configures CPU affinity, and creates the session
    bool Initialize(const std::string& model_path, const QnnEPConfig& qnn_config, const CpuAffinityConfig& cpu_config);

    // Executes acoustic model inference on raw PCM float audio (16kHz sample rate)
    // Decodes speech to text tokens and outputs the transcribed string
    std::string Transcribe(const std::vector<float>& pcm_data);

    // Explicit thread affinity binding helper targeting Snapdragon Oryon/Kryo CPU cluster configurations
    static bool SetThreadAffinity(const CpuAffinityConfig& config);

    // Sets the calling thread's scheduler to SCHED_FIFO with the given priority
    static bool SetAudioThreadPriority(int priority);

    // Returns latency metrics from the last inference run in milliseconds
    float GetLastInferenceLatencyMs() const { return last_inference_latency_ms_; }
    float GetFeatureExtractionLatencyMs() const { return feat_extraction_latency_ms_; }
    float GetNpuInferenceLatencyMs() const { return npu_inference_latency_ms_; }

private:
    // Extract Log-Mel Spectrogram features from raw audio buffer
    std::vector<float> ExtractFeatures(const std::vector<float>& pcm_data, int& out_seq_len, int& out_feature_dim);

    // ONNX Runtime Core Objects
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    // Model structures
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> input_dims_;
    std::vector<int64_t> output_dims_;

    // Performance measurements
    float last_inference_latency_ms_ = 0.0f;
    float feat_extraction_latency_ms_ = 0.0f;
    float npu_inference_latency_ms_ = 0.0f;

    // ASR Vocabulary mapper for decoding token IDs to subwords/characters
    std::string vocab_file_path_;
    std::vector<std::string> vocabulary_;
    void LoadVocabulary(const std::string& vocab_path);
    std::string GreedyDecode(const std::vector<int64_t>& token_ids);

    // Cached Mel filterbank coefficients for fast feature extraction
    struct MelFilterbank {
        bool initialized = false;
        int num_mels = 80;
        int fft_size = 512;
        int sample_rate = 16000;
        std::vector<std::vector<float>> weights; // [num_mels, fft_size / 2 + 1]
    } mel_fb_;
    void InitializeMelFilters();

    bool is_initialized_ = false;
};

} // namespace inference
} // namespace edgedeploy
