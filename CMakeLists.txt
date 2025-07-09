cmake_minimum_required(VERSION 3.18)
project(GPU_IPC LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Set CUDA architecture (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Example 1: Basic Shared Memory
add_executable(basic_producer examples/01_basic_shared_memory/producer.cu)
add_executable(basic_consumer examples/01_basic_shared_memory/consumer.cu)
set_property(TARGET basic_producer basic_consumer PROPERTY CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(basic_producer ${CUDA_LIBRARIES})
target_link_libraries(basic_consumer ${CUDA_LIBRARIES})

# Example 2: Optimized Shared Memory
add_executable(optimized_producer examples/02_optimized_shared_memory/producer.cu)
add_executable(optimized_consumer examples/02_optimized_shared_memory/consumer.cu)
set_property(TARGET optimized_producer optimized_consumer PROPERTY CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(optimized_producer ${CUDA_LIBRARIES})
target_link_libraries(optimized_consumer ${CUDA_LIBRARIES})

# Example 3: CUDA IPC with Sleep
add_executable(ipc_sleep_producer examples/03_cuda_ipc_sleep/producer.cu)
add_executable(ipc_sleep_consumer examples/03_cuda_ipc_sleep/consumer.cu)
set_property(TARGET ipc_sleep_producer ipc_sleep_consumer PROPERTY CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(ipc_sleep_producer ${CUDA_LIBRARIES})
target_link_libraries(ipc_sleep_consumer ${CUDA_LIBRARIES})

# Example 4: CUDA IPC with Spin-Wait (World Record)
add_executable(ipc_spinwait_producer examples/04_cuda_ipc_spinwait/producer.cu)
add_executable(ipc_spinwait_consumer examples/04_cuda_ipc_spinwait/consumer.cu)
set_property(TARGET ipc_spinwait_producer ipc_spinwait_consumer PROPERTY CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(ipc_spinwait_producer ${CUDA_LIBRARIES})
target_link_libraries(ipc_spinwait_consumer ${CUDA_LIBRARIES})

# Benchmarks
add_executable(gpu_limits_benchmark benchmarks/gpu_limits_benchmark.cu)
add_executable(memory_benchmark benchmarks/memory_benchmark.cu)
add_executable(cuda_ipc_diagnostic benchmarks/cuda_ipc_diagnostic.cu)
add_executable(simpleIPC benchmarks/simpleIPC.cu)

set_property(TARGET gpu_limits_benchmark memory_benchmark cuda_ipc_diagnostic simpleIPC PROPERTY CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(gpu_limits_benchmark ${CUDA_LIBRARIES})
target_link_libraries(memory_benchmark ${CUDA_LIBRARIES})
target_link_libraries(cuda_ipc_diagnostic ${CUDA_LIBRARIES})
target_link_libraries(simpleIPC ${CUDA_LIBRARIES})

# Windows-specific settings
if(WIN32)
    set(ALL_TARGETS 
        basic_producer basic_consumer
        optimized_producer optimized_consumer
        ipc_sleep_producer ipc_sleep_consumer
        ipc_spinwait_producer ipc_spinwait_consumer
        gpu_limits_benchmark memory_benchmark cuda_ipc_diagnostic simpleIPC
    )
    
    foreach(target ${ALL_TARGETS})
        target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN)
    endforeach()
endif()

# Set runtime directory
set_target_properties(${ALL_TARGETS} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Create convenience targets
add_custom_target(examples)
add_dependencies(examples 
    basic_producer basic_consumer
    optimized_producer optimized_consumer
    ipc_sleep_producer ipc_sleep_consumer
    ipc_spinwait_producer ipc_spinwait_consumer
)

add_custom_target(benchmarks_target)
add_dependencies(benchmarks_target
    gpu_limits_benchmark memory_benchmark cuda_ipc_diagnostic simpleIPC
)

# Install targets
install(TARGETS ${ALL_TARGETS} DESTINATION bin)
install(FILES 
    README.md
    GPU_IPC_Research_Report.md
    DESTINATION .
)
install(DIRECTORY examples/ DESTINATION examples)
install(DIRECTORY docs/ DESTINATION docs)
install(DIRECTORY include/ DESTINATION include)