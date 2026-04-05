cmake_minimum_required(VERSION 3.18)
project(EdgeDeployASR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure directories
set(ASR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ASR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Define the shared library target
add_library(edgedeploy_asr SHARED
    ${ASR_SRC_DIR}/asr_engine.cpp
    ${ASR_SRC_DIR}/jni_bridge.cpp
)

# Header search paths
target_include_directories(edgedeploy_asr PRIVATE
    ${ASR_INCLUDE_DIR}
)

# Search for ONNX Runtime library
# On Android, the ONNX Runtime dynamic library (libonnxruntime.so) is usually bundled
# in the JNI libs directories. We reference it by importing.
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../android/libs/${ANDROID_ABI}/libonnxruntime.so"
)

# Link log library (needed for JNI Logging) and ONNX Runtime
find_library(log-lib log)
target_link_libraries(edgedeploy_asr
    onnxruntime
    ${log-lib}
)

# Optimization flags targeting Snapdragon Oryon/Kryo arm64-v8a cores
if(ANDROID)
    target_compile_options(edgedeploy_asr PRIVATE
        -O3
        -march=armv8-a+simd
        -ffast-math
        -flto
    )
    target_link_options(edgedeploy_asr PRIVATE
        -flto
        -Wl,--gc-sections
    )
endif()
