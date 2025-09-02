cmake_minimum_required(VERSION 3.15)
project(HyperionEngineProject CXX)

# --- Standard C++ Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
# The Linux LibTorch download uses the modern C++ ABI
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# --- Define the path to your LibTorch directory for convenience ---
set(LIBTORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch)

# --- Compiler Flags for Our Code ---
if(NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -O3)
endif()

# --- Output Directory ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Collect ALL Engine Source Files in ONE list ---
set(HYPERION_SOURCES
    src/cpp/main.cpp
    src/cpp/core/bitboard.cpp
    src/cpp/core/movegen.cpp
    src/cpp/core/position.cpp
    src/cpp/core/zobrist.cpp
    src/cpp/search/eval.cpp
    src/cpp/search/search.cpp
    src/cpp/search/tt.cpp
    src/cpp/search/move_encoder.cpp
    src/cpp/nn_inference/nn_inference.cpp
    src/cpp/uci/engine.cpp
    src/cpp/uci/uci.cpp
)

# --- Main Hyperion Engine Executable ---
add_executable(HyperionEngine ${HYPERION_SOURCES})

# --- MANUAL INCLUDE PATH for LibTorch ---
# Add all necessary include directories for the executable.
target_include_directories(HyperionEngine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    SYSTEM ${LIBTORCH_DIR}/include
    SYSTEM ${LIBTORCH_DIR}/include/torch/csrc/api/include
)

# --- Suppress Warnings from LibTorch Headers ---
if(NOT MSVC)
    target_compile_options(HyperionEngine PRIVATE
        -Wno-pedantic
        -Wno-unused-parameter
        -Wno-dangling-reference
    )
endif()

# --- MANUAL LINKING for the Linux LibTorch (.so files) ---
# Link the executable against the necessary Torch libraries.
# The order can be important. Generally, link higher-level libraries first.
target_link_libraries(HyperionEngine PRIVATE
    # The main library with the JIT frontend and high-level API.
    ${LIBTORCH_DIR}/lib/libtorch.so

    # The CUDA backend for GPU operations.
    ${LIBTORCH_DIR}/lib/libtorch_cuda.so

    # The core CUDA tensor library. libtorch_cuda.so depends on this.
    ${LIBTORCH_DIR}/lib/libc10_cuda.so

    # The CPU backend is REQUIRED. Both libtorch.so and libtorch_cuda.so
    # have dependencies on symbols within this library. Removing it
    # will cause linker errors.
    ${LIBTORCH_DIR}/lib/libtorch_cpu.so
    
    # The core C10 tensor library (CPU part). All other libs depend on this.
    ${LIBTORCH_DIR}/lib/libc10.so
)

# --- BMI2 OPTION ---
option(HYPERION_ENABLE_BMI2 "Enable BMI2 optimizations" ON)
if(HYPERION_ENABLE_BMI2 AND NOT MSVC)
    target_compile_options(HyperionEngine PRIVATE -mbmi2)
endif()