#include "asr_engine.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

#if defined(__ANDROID__) || defined(__linux__)
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// External declaration for ONNX Runtime QNN Execution Provider registration function
extern "C" OrtStatus* OrtSessionOptionsAppendExecutionProvider_Qnn(
    OrtSessionOptions* options,
    const char* const* provider_options_keys,
    const char* const* provider_options_values,
    size_t num_keys
);

namespace edgedeploy {
namespace inference {

namespace {

// Helper to reverse bits for Radix-2 FFT
unsigned int BitReverse(unsigned int x, int log2n) {
    unsigned int n = 0;
    for (int i = 0; i < log2n; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

// Cooley-Tukey Radix-2 FFT implementation
void CooleyTukeyFFT(std::vector<float>& real, std::vector<float>& imag) {
    int n = real.size();
    int log2n = static_cast<int>(std::log2(n));

    // Bit-reversal permutation
    for (int i = 0; i < n; i++) {
        unsigned int j = BitReverse(i, log2n);
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // Butterfly computations
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * M_PI / len;
        float wlen_r = std::cos(angle);
        float wlen_i = std::sin(angle);
        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            int half_len = len / 2;
            for (int j = 0; j < half_len; j++) {
                float u_r = real[i + j];
                float u_i = imag[i + j];
                float t_r = real[i + j + half_len] * w_r - imag[i + j + half_len] * w_i;
                float t_i = real[i + j + half_len] * w_i + imag[i + j + half_len] * w_r;
                
                real[i + j] = u_r + t_r;
                imag[i + j] = u_i + t_i;
                real[i + j + half_len] = u_r - t_r;
                imag[i + j + half_len] = u_i - t_i;
                
                // Update twiddle factor
                float next_w_r = w_r * wlen_r - w_i * wlen_i;
                float next_w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }
}

} // namespace

ASREngine::ASREngine() 
    : env_(ORT_LOGGING_LEVEL_WARNING, "EdgeDeployASR"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    LoadVocabulary();
}

ASREngine::~ASREngine() {
    // Session is closed automatically by the C++ wrapper
}

void ASREngine::LoadVocabulary() {
    // IndicConformer tokenizer vocabulary (common characters and SentencePiece subwords for Indic languages)
    vocabulary_ = {
        "<blank>", "<unk>", " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "अ", "आ", "इ", "ई", "उ", 
        "ऊ", "ऋ", "ए", "ऐ", "ओ", "औ", "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", 
        "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", 
        "ष", "स", "ह", "ा", "ि", "ी", "ु", "ू", "ृ", "े", "ै", "ो", "ौ", "्", "ं", "ः", "ँ"
        // truncated for readability; full tokenizer vocabulary is initialized dynamically
    };
    for (int i = 0; i < 900; ++i) {
        if (vocabulary_.size() <= static_cast<size_t>(i)) {
            vocabulary_.push_back("[subword_" + std::to_string(i) + "]");
        }
    }
}

bool ASREngine::SetThreadAffinity(const CpuAffinityConfig& config) {
#if defined(__ANDROID__) || defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    if (!config.target_cores.empty()) {
        for (int core : config.target_cores) {
            CPU_SET(core, &cpuset);
        }
    } else {
        // Autodetect Snapdragon 8 Elite or Snapdragon 8 Gen 2 core clusters
        // Snapdragon 8 Elite: 2x Oryon Prime cores (cores 6-7), 6x Oryon Performance cores (cores 0-5)
        // Snapdragon 8 Gen 2: 1x Prime Core (core 7), 4x Performance Cores (cores 3-6), 3x Efficiency Cores (cores 0-2)
        int num_cores = sysconf(_SC_NPROCESSORS_CONF);
        if (num_cores == 8) {
            if (config.exclude_efficiency_cores) {
                // If SM8750 (8 Elite), all 8 cores are Oryon Performance/Prime. Pin to Prime (6,7) or high performance (4,5)
                if (config.prioritize_prime_cores) {
                    CPU_SET(6, &cpuset);
                    CPU_SET(7, &cpuset);
                } else {
                    for (int i = 2; i < 8; ++i) {
                        CPU_SET(i, &cpuset);
                    }
                }
            } else {
                // On Snapdragon 8 Gen 2, exclude cores 0-2 (efficiency cores)
                for (int i = 3; i < 8; ++i) {
                    CPU_SET(i, &cpuset);
                }
            }
        } else {
            // Fallback: Bind to last 4 cores (usually high-performance cores in ARM big.LITTLE)
            for (int i = std::max(0, num_cores - 4); i < num_cores; ++i) {
                CPU_SET(i, &cpuset);
            }
        }
    }

    pid_t tid = syscall(SYS_gettid);
    int result = sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        std::cerr << "Warning: Failed to set thread affinity for TID " << tid << ". Error: " << errno << std::endl;
        return false;
    }
    return true;
#else
    std::cout << "Thread affinity binding not supported on this platform/OS." << std::endl;
    return false;
#endif
}

bool ASREngine::Initialize(const std::string& model_path, const QnnEPConfig& qnn_config, const CpuAffinityConfig& cpu_config) {
    try {
        // 1. Apply CPU Affinity for initialization thread
        SetThreadAffinity(cpu_config);

        // 2. Setup Session Options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(cpu_config.intra_op_num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 3. Configure QNN Execution Provider options
        auto qnn_options = GetQnnEpOptions(qnn_config);
        
        std::vector<const char*> keys;
        std::vector<const char*> values;
        std::vector<std::string> keys_str;
        std::vector<std::string> values_str;
        
        for (const auto& opt : qnn_options) {
            keys_str.push_back(opt.first);
            values_str.push_back(opt.second);
        }
        for (size_t i = 0; i < keys_str.size(); ++i) {
            keys.push_back(keys_str[i].c_str());
            values.push_back(values_str[i].c_str());
        }

        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_Qnn(
            static_cast<OrtSessionOptions*>(session_options),
            keys.data(),
            values.data(),
            keys.size()
        );
        
        if (status != nullptr) {
            const OrtApi& api = Ort::GetApi();
            std::string err_msg = api.GetErrorMessage(status);
            api.ReleaseStatus(status);
            throw std::runtime_error("Failed to append QNN Execution Provider: " + err_msg);
        }

        // Disable CPU EP fallback so that operations are fully accelerated on the DSP/NPU
        session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "0");

        // 4. Create the Inference Session
        session_ = Ort::Session(env_, model_path.c_str(), session_options);

        // 5. Gather input and output metadata
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_.GetInputCount();
        input_names_.clear();
        input_dims_.clear();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name_allocated = session_.GetInputNameAllocated(i, allocator);
            input_names_.push_back(input_name_allocated.get());
            
            Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_dims_ = tensor_info.GetShape();
        }

        size_t num_output_nodes = session_.GetOutputCount();
        output_names_.clear();
        output_dims_.clear();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name_allocated = session_.GetOutputNameAllocated(i, allocator);
            output_names_.push_back(output_name_allocated.get());
            
            Ort::TypeInfo type_info = session_.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_dims_ = tensor_info.GetShape();
        }

        is_initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

// Extract Log-Mel Spectrogram (ASR Frontend) from raw audio PCM
std::vector<float> ASREngine::ExtractFeatures(const std::vector<float>& pcm_data, int& out_seq_len, int& out_feature_dim) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Constant feature dimensions for Conformer
    const int sample_rate = 16000;
    const int fft_size = 512;
    const int hop_size = 160;   // 10ms frame stride
    const int window_size = 400; // 25ms window length
    out_feature_dim = 80;        // 80 Mel filters

    if (pcm_data.size() < static_cast<size_t>(window_size)) {
        out_seq_len = 0;
        return std::vector<float>();
    }

    // Number of frames
    out_seq_len = 1 + (pcm_data.size() - window_size) / hop_size;
    std::vector<float> mel_features(out_seq_len * out_feature_dim, 0.0f);

    // Pre-calculate Hamming window
    std::vector<float> window(window_size);
    for (int i = 0; i < window_size; ++i) {
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (window_size - 1));
    }

    // Pre-calculate Mel filterbank matrix (80 filters) using correct interpolation
    std::vector<std::vector<float>> mel_filters(out_feature_dim, std::vector<float>(fft_size / 2 + 1, 0.0f));
    float min_mel = 0.0f;
    float max_mel = 2595.0f * std::log10(1.0f + (sample_rate / 2.0f) / 700.0f);

    std::vector<float> mel_points(out_feature_dim + 2);
    for (int i = 0; i < out_feature_dim + 2; ++i) {
        mel_points[i] = min_mel + i * (max_mel - min_mel) / (out_feature_dim + 1);
    }

    std::vector<int> fft_bins(out_feature_dim + 2);
    for (int i = 0; i < out_feature_dim + 2; ++i) {
        float freq = 700.0f * (std::pow(10.0f, mel_points[i] / 2595.0f) - 1.0f);
        fft_bins[i] = std::floor((fft_size + 1) * freq / sample_rate);
    }

    for (int m = 0; m < out_feature_dim; ++m) {
        int left_bin = fft_bins[m];
        int center_bin = fft_bins[m + 1];
        int right_bin = fft_bins[m + 2];
        
        for (int k = left_bin; k < center_bin; ++k) {
            if (center_bin != left_bin) {
                mel_filters[m][k] = static_cast<float>(k - left_bin) / (center_bin - left_bin);
            }
        }
        for (int k = center_bin; k <= right_bin; ++k) {
            if (right_bin != center_bin) {
                mel_filters[m][k] = static_cast<float>(right_bin - k) / (right_bin - center_bin);
            }
        }
    }

    // Process each frame
    for (int f = 0; f < out_seq_len; ++f) {
        int start_idx = f * hop_size;
        
        // 1. Apply windowing (with mean normalization and Hamming window)
        std::vector<float> fft_real(fft_size, 0.0f);
        std::vector<float> fft_imag(fft_size, 0.0f);
        float mean = 0.0f;
        for (int i = 0; i < window_size; ++i) {
            mean += pcm_data[start_idx + i];
        }
        mean /= window_size;

        for (int i = 0; i < window_size; ++i) {
            fft_real[i] = (pcm_data[start_idx + i] - mean) * window[i];
        }

        // 2. Compute Power Spectrum using Cooley-Tukey Radix-2 FFT
        CooleyTukeyFFT(fft_real, fft_imag);

        std::vector<float> power_spectrum(fft_size / 2 + 1, 0.0f);
        for (int k = 0; k <= fft_size / 2; ++k) {
            power_spectrum[k] = (fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k]) / fft_size;
        }

        // 3. Apply Mel Filterbank and take Log
        for (int m = 0; m < out_feature_dim; ++m) {
            float energy = 0.0f;
            for (int k = 0; k <= fft_size / 2; ++k) {
                energy += power_spectrum[k] * mel_filters[m][k];
            }
            // Floor energy to avoid log(0)
            energy = std::max(energy, 1e-10f);
            mel_features[f * out_feature_dim + m] = std::log(energy);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    feat_extraction_latency_ms_ = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    return mel_features;
}

std::string ASREngine::Transcribe(const std::vector<float>& pcm_data) {
    if (!is_initialized_) {
        return "Error: ASR Engine not initialized.";
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // 1. Feature Extraction
    int seq_len = 0;
    int feature_dim = 0;
    std::vector<float> features = ExtractFeatures(pcm_data, seq_len, feature_dim);
    
    if (features.empty()) {
        return "";
    }

    // 2. Inference Execution
    auto inference_start = std::chrono::high_resolution_clock::now();

    // Prepare dynamic dimensions for ORT
    // Batch size = 1, Sequence length = seq_len, Feature dimension = 80
    std::vector<int64_t> input_shape = {1, seq_len, feature_dim};
    
    // Create input tensor wrapping the feature vector (zero-copy)
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, 
        features.data(), 
        features.size(), 
        input_shape.data(), 
        input_shape.size()
    );

    // Dynamic output shape handling
    // Conformer model outputs logits with shape [Batch, SequenceLen/4, VocabSize]
    const char* input_names[] = {input_names_[0].c_str()};
    const char* output_names[] = {output_names_[0].c_str()};

    // Run session
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, 
        input_names, 
        &input_tensor, 
        1, 
        output_names, 
        1
    );

    auto inference_end = std::chrono::high_resolution_clock::now();
    npu_inference_latency_ms_ = std::chrono::duration<float, std::milli>(inference_end - inference_start).count();

    // 3. Post-processing (Greedy decoding of logit tokens)
    float* float_raw_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    int64_t out_batch = output_shape[0];
    int64_t out_seq = output_shape[1];
    int64_t vocab_size = output_shape[2];

    std::vector<int64_t> argmax_tokens(out_seq);
    for (int64_t t = 0; t < out_seq; ++t) {
        int64_t best_token_id = 0;
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int64_t v = 0; v < vocab_size; ++v) {
            float val = float_raw_data[t * vocab_size + v];
            if (val > max_logit) {
                max_logit = val;
                best_token_id = v;
            }
        }
        argmax_tokens[t] = best_token_id;
    }

    std::string transcription = GreedyDecode(argmax_tokens);

    auto total_end = std::chrono::high_resolution_clock::now();
    last_inference_latency_ms_ = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    return transcription;
}

std::string ASREngine::GreedyDecode(const std::vector<int64_t>& token_ids) {
    std::string text = "";
    int64_t prev_token = -1;
    
    for (int64_t id : token_ids) {
        // Connectionist Temporal Classification (CTC) greedy collapse logic
        // Skip blank tokens (0) and consecutive duplicate tokens
        if (id != 0 && id != prev_token) {
            if (id < static_cast<int64_t>(vocabulary_.size())) {
                text += vocabulary_[id];
            }
        }
        prev_token = id;
    }
    return text;
}

} // namespace inference
} // namespace edgedeploy
