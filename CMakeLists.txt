cmake_minimum_required(VERSION 3.15)
project(HyperionEngineProject CXX)

#Standard C++ Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
set(LIBTORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch)
if(NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -O3)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
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

add_executable(HyperionEngine ${HYPERION_SOURCES})

target_include_directories(HyperionEngine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    SYSTEM ${LIBTORCH_DIR}/include
    SYSTEM ${LIBTORCH_DIR}/include/torch/csrc/api/include
)

if(NOT MSVC)
    target_compile_options(HyperionEngine PRIVATE
        -Wno-pedantic
        -Wno-unused-parameter
        -Wno-dangling-reference
    )
endif()

target_link_libraries(HyperionEngine PRIVATE
    ${LIBTORCH_DIR}/lib/libtorch.so

    ${LIBTORCH_DIR}/lib/libtorch_cuda.so

    ${LIBTORCH_DIR}/lib/libc10_cuda.so
    ${LIBTORCH_DIR}/lib/libtorch_cpu.so
    
    ${LIBTORCH_DIR}/lib/libc10.so
)

option(HYPERION_ENABLE_BMI2 "Enable BMI2 optimizations" ON)
if(HYPERION_ENABLE_BMI2 AND NOT MSVC)
    target_compile_options(HyperionEngine PRIVATE -mbmi2)
endif()