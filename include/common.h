#pragma once

#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>

// 4K image dimensions
constexpr int IMAGE_WIDTH = 3840;
constexpr int IMAGE_HEIGHT = 2160;
constexpr int IMAGE_CHANNELS = 3; // RGB
constexpr size_t IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS;

// Shared memory structure
struct SharedImageData {
    HANDLE ipcHandle;
    void* d_imageData;
    size_t imageSize;
    int frameNumber;
};

// Named pipe for IPC handle sharing
constexpr const char* PIPE_NAME = "\\\\.\\pipe\\cuda_ipc_pipe";

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)