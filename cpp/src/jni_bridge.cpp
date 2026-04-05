#include <jni.h>
#include <string>
#include <vector>
#include "asr_engine.h"

using namespace edgedeploy::inference;

// Helper to convert JNI string to std::string
std::string jstring2string(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* str = env->GetStringUTFChars(jstr, nullptr);
    std::string std_str(str);
    env->ReleaseStringUTFChars(jstr, str);
    return std_str;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_edgedeploy_inference_ASRManager_nativeInit(
    JNIEnv* env,
    jobject thiz,
    jstring model_path,
    jstring backend_path,
    jint performance_profile, // 0 = LOW_POWER, 1 = BALANCED, 2 = HIGH, 3 = BURST
    jint precision_type,       // 0 = FP32, 1 = FP16, 2 = INT8_DYNAMIC
    jboolean use_htp_fp16,
    jboolean enable_context_caching,
    jstring cache_dir,
    jintArray cpu_cores,
    jint intra_threads,
    jstring vocab_path
) {
    ASREngine* engine = new ASREngine();
    
    QnnEPConfig qnn_config;
    qnn_config.backend_path = jstring2string(env, backend_path);
    qnn_config.use_htp_fp16_precision = use_htp_fp16;
    qnn_config.enable_context_caching = enable_context_caching;
    qnn_config.context_cache_dir = jstring2string(env, cache_dir);
    qnn_config.vocab_file_path = jstring2string(env, vocab_path);
    
    switch (performance_profile) {
        case 0: qnn_config.perf_profile = PerformanceProfile::LOW_POWER; break;
        case 1: qnn_config.perf_profile = PerformanceProfile::BALANCED; break;
        case 2: qnn_config.perf_profile = PerformanceProfile::HIGH_PERFORMANCE; break;
        case 3: qnn_config.perf_profile = PerformanceProfile::BURST; break;
        default: qnn_config.perf_profile = PerformanceProfile::BURST; break;
    }
    
    switch (precision_type) {
        case 0: qnn_config.precision = NpuPrecision::FP32; break;
        case 1: qnn_config.precision = NpuPrecision::FP16; break;
        case 2: qnn_config.precision = NpuPrecision::INT8_DYNAMIC; break;
        default: qnn_config.precision = NpuPrecision::INT8_DYNAMIC; break;
    }
    
    CpuAffinityConfig cpu_config;
    cpu_config.intra_op_num_threads = intra_threads;
    
    if (cpu_cores != nullptr) {
        jsize len = env->GetArrayLength(cpu_cores);
        jint* body = env->GetIntArrayElements(cpu_cores, 0);
        for (int i = 0; i < len; i++) {
            cpu_config.target_cores.push_back(body[i]);
        }
        env->ReleaseIntArrayElements(cpu_cores, body, 0);
    }
    
    std::string model_path_str = jstring2string(env, model_path);
    bool success = engine->Initialize(model_path_str, qnn_config, cpu_config);
    
    if (!success) {
        delete engine;
        return 0;
    }
    
    return reinterpret_cast<jlong>(engine);
}

JNIEXPORT jstring JNICALL
Java_com_edgedeploy_inference_ASRManager_nativeTranscribe(
    JNIEnv* env,
    jobject thiz,
    jlong engine_handle,
    jfloatArray audio_data
) {
    ASREngine* engine = reinterpret_cast<ASREngine*>(engine_handle);
    if (!engine) {
        return env->NewStringUTF("Error: Engine handle is null");
    }
    
    jsize len = env->GetArrayLength(audio_data);
    jfloat* body = env->GetFloatArrayElements(audio_data, 0);
    
    std::vector<float> pcm_data(body, body + len);
    env->ReleaseFloatArrayElements(audio_data, body, JNI_ABORT);
    
    std::string text = engine->Transcribe(pcm_data);
    return env->NewStringUTF(text.c_str());
}

JNIEXPORT jfloatArray JNICALL
Java_com_edgedeploy_inference_ASRManager_nativeGetLatencies(
    JNIEnv* env,
    jobject thiz,
    jlong engine_handle
) {
    ASREngine* engine = reinterpret_cast<ASREngine*>(engine_handle);
    jfloatArray result = env->NewFloatArray(3);
    if (!engine) {
        return result;
    }
    
    float latencies[3] = {
        engine->GetLastInferenceLatencyMs(),
        engine->GetFeatureExtractionLatencyMs(),
        engine->GetNpuInferenceLatencyMs()
    };
    
    env->SetFloatArrayRegion(result, 0, 3, latencies);
    return result;
}

JNIEXPORT void JNICALL
Java_com_edgedeploy_inference_ASRManager_nativeRelease(
    JNIEnv* env,
    jobject thiz,
    jlong engine_handle
) {
    ASREngine* engine = reinterpret_cast<ASREngine*>(engine_handle);
    if (engine) {
        delete engine;
    }
}

} // extern "C"
