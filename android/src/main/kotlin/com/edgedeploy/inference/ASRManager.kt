package com.edgedeploy.inference

import android.util.Log
import java.io.Closeable

class ASRManager(
    private val modelPath: String,
    private val config: Config = Config()
) : Closeable {

    companion object {
        private const val TAG = "ASRManager"

        init {
            try {
                System.loadLibrary("onnxruntime")
                System.loadLibrary("edgedeploy_asr")
                Log.i(TAG, "Native libraries loaded successfully.")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native libraries: ${e.message}")
            }
        }
    }

    enum class PerformanceProfile(val value: Int) {
        LOW_POWER(0),
        BALANCED(1),
        HIGH_PERFORMANCE(2),
        BURST(3)
    }

    enum class PrecisionType(val value: Int) {
        FP32(0),
        FP16(1),
        INT8_DYNAMIC(2)
    }

    data class Config(
        val qnnBackendPath: String = "libQnnHtp.so",
        val performanceProfile: PerformanceProfile = PerformanceProfile.BURST,
        val precisionType: PrecisionType = PrecisionType.INT8_DYNAMIC,
        val useHtpFp16: Boolean = true,
        val enableContextCaching: Boolean = true,
        val contextCacheDir: String = "/data/local/tmp/qnn_context_cache",
        val targetCpuCores: IntArray? = intArrayOf(6, 7), // Snapdragon 8 Elite Prime cores by default
        val intraThreadsCount: Int = 4,
        val vocabPath: String = ""
    )

    private var nativeEngineHandle: Long = 0

    init {
        nativeEngineHandle = nativeInit(
            modelPath = modelPath,
            backendPath = config.qnnBackendPath,
            performanceProfile = config.performanceProfile.value,
            precisionType = config.precisionType.value,
            useHtpFp16 = config.useHtpFp16,
            enableContextCaching = config.enableContextCaching,
            cacheDir = config.contextCacheDir,
            cpuCores = config.targetCpuCores,
            intraThreads = config.intraThreadsCount,
            vocabPath = config.vocabPath
        )
        if (nativeEngineHandle == 0L) {
            throw IllegalStateException("Failed to initialize native ASR Engine.")
        }
        Log.d(TAG, "ASR Engine successfully initialized on handle: $nativeEngineHandle")
    }

    /**
     * Transcribes a raw 16kHz PCM audio float array.
     * @param audioData FloatArray containing normalized audio samples (-1.0 to 1.0)
     * @return Transcribed text string
     */
    fun transcribe(audioData: FloatArray): String {
        checkInitialized()
        return nativeTranscribe(nativeEngineHandle, audioData)
    }

    /**
     * Retrieves latency statistics of the last transcription execution.
     * Index 0: Total latency (ms)
     * Index 1: Mel-Spectrogram Feature extraction latency (ms)
     * Index 2: Hexagon NPU QNN EP Inference latency (ms)
     */
    fun getLatencies(): LatencyMetrics {
        checkInitialized()
        val latencies = nativeGetLatencies(nativeEngineHandle)
        return LatencyMetrics(
            totalLatencyMs = latencies.getOrElse(0) { 0.0f },
            featExtractionMs = latencies.getOrElse(1) { 0.0f },
            npuInferenceMs = latencies.getOrElse(2) { 0.0f }
        )
    }

    override fun close() {
        if (nativeEngineHandle != 0L) {
            nativeRelease(nativeEngineHandle)
            Log.d(TAG, "ASR Engine handle $nativeEngineHandle released.")
            nativeEngineHandle = 0L
        }
    }

    private fun checkInitialized() {
        if (nativeEngineHandle == 0L) {
            throw IllegalStateException("ASR Engine has been closed or was not initialized.")
        }
    }

    data class LatencyMetrics(
        val totalLatencyMs: Float,
        val featExtractionMs: Float,
        val npuInferenceMs: Float
    )

    // JNI Declarations
    private native fun nativeInit(
        modelPath: String,
        backendPath: String,
        performanceProfile: Int,
        precisionType: Int,
        useHtpFp16: Boolean,
        enableContextCaching: Boolean,
        cacheDir: String,
        cpuCores: IntArray?,
        intraThreads: Int,
        vocabPath: String
    ): Long

    private native fun nativeTranscribe(engineHandle: Long, audioData: FloatArray): String
    private native fun nativeGetLatencies(engineHandle: Long): FloatArray
    private native fun nativeRelease(engineHandle: Long)
}
