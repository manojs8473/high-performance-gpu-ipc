#include "common.h"
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <windows.h>

__global__ void generateImage(unsigned char* imageData, int frameNumber, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        int pixelIdx = (idy * width + idx) * 3;
        
        // Generate a simple animated pattern
        imageData[pixelIdx] = (idx + frameNumber) % 256;     // R
        imageData[pixelIdx + 1] = (idy + frameNumber) % 256; // G
        imageData[pixelIdx + 2] = (idx + idy + frameNumber) % 256; // B
    }
}

struct IPCHeader {
    cudaIpcMemHandle_t memHandle;
    int frameNumber;
    bool dataReady;
    bool processComplete;
    size_t imageSize;
    bool consumerReady;
};

int main() {
    std::cout << "Working IPC Producer: Starting..." << std::endl;
    
    // Initialize CUDA with proper checks (like NVIDIA sample)
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return 1;
    }
    
    // Check device properties like NVIDIA sample
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    
    if (!prop.unifiedAddressing) {
        std::cerr << "Device does not support unified addressing" << std::endl;
        return 1;
    }
    
    if (prop.computeMode != cudaComputeModeDefault) {
        std::cerr << "Device is in unsupported compute mode" << std::endl;
        return 1;
    }
    
    checkCudaErrors(cudaSetDevice(0));
    
    std::cout << "Working IPC Producer: Device checks passed" << std::endl;
    
    // Create shared memory for header
    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(IPCHeader),
        "Local\\WorkingIPCHeader"
    );
    
    if (hMapFile == NULL) {
        std::cerr << "Could not create header mapping: " << GetLastError() << std::endl;
        return 1;
    }
    
    void* pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(IPCHeader));
    if (pBuf == NULL) {
        std::cerr << "Could not map header view: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return 1;
    }
    
    IPCHeader* header = (IPCHeader*)pBuf;
    memset(header, 0, sizeof(IPCHeader));
    
    // Allocate GPU memory for IPC (like NVIDIA sample)
    unsigned char* d_imageData;
    checkCudaErrors(cudaMalloc(&d_imageData, IMAGE_SIZE));
    
    // Get IPC handle (using NVIDIA's approach)
    checkCudaErrors(cudaIpcGetMemHandle(&header->memHandle, d_imageData));
    
    header->imageSize = IMAGE_SIZE;
    header->frameNumber = 0;
    header->dataReady = false;
    header->processComplete = true;
    header->consumerReady = false;
    
    std::cout << "Working IPC Producer: IPC handle created successfully" << std::endl;
    
    // Setup CUDA kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Working IPC Producer: Waiting for consumer..." << std::endl;
    
    // Wait for consumer to connect
    while (!header->consumerReady) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::cout << "Working IPC Producer: Starting GPU IPC image generation..." << std::endl;
    
    int frameNumber = 0;
    auto lastTime = std::chrono::high_resolution_clock::now();
    int fpsCounter = 0;
    
    while (true) {
        // Wait for previous frame to be processed
        while (!header->processComplete) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        
        // Generate image directly on shared GPU memory (TRUE ZERO-COPY!)
        generateImage<<<gridSize, blockSize>>>(d_imageData, frameNumber, IMAGE_WIDTH, IMAGE_HEIGHT);
        checkCudaErrors(cudaDeviceSynchronize());
        
        // Update header - NO MEMORY TRANSFERS NEEDED!
        header->frameNumber = frameNumber;
        header->processComplete = false;
        header->dataReady = true;
        
        frameNumber++;
        fpsCounter++;
        
        // FPS calculation
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);
        if (duration.count() > 1000) {
            std::cout << "Working IPC Producer FPS: " << (fpsCounter * 1000.0 / duration.count()) << std::endl;
            lastTime = currentTime;
            fpsCounter = 0;
        }
    }
    
    // Cleanup
    cudaFree(d_imageData);
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    
    return 0;
}