#include "common.h"
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <windows.h>

__global__ void processImage(unsigned char* imageData, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        int pixelIdx = (idy * width + idx) * 3;
        
        // Simple processing: invert colors
        imageData[pixelIdx] = 255 - imageData[pixelIdx];
        imageData[pixelIdx + 1] = 255 - imageData[pixelIdx + 1];
        imageData[pixelIdx + 2] = 255 - imageData[pixelIdx + 2];
    }
}

struct IPCHeader {
    cudaIpcMemHandle_t memHandle;
    volatile int frameNumber;
    volatile bool dataReady;
    volatile bool processComplete;
    size_t imageSize;
    volatile bool consumerReady;
};

int main() {
    std::cout << "Spin-Wait IPC Consumer: Starting..." << std::endl;
    
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
    
    std::cout << "Spin-Wait IPC Consumer: Device checks passed" << std::endl;
    
    // Open shared memory for header
    HANDLE hMapFile = OpenFileMapping(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        "Local\\SpinWaitIPCHeader"
    );
    
    if (hMapFile == NULL) {
        std::cerr << "Could not open header mapping: " << GetLastError() << std::endl;
        return 1;
    }
    
    void* pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(IPCHeader));
    if (pBuf == NULL) {
        std::cerr << "Could not map header view: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return 1;
    }
    
    IPCHeader* header = (IPCHeader*)pBuf;
    
    // Signal that consumer is ready
    header->consumerReady = true;
    
    std::cout << "Spin-Wait IPC Consumer: Signaled ready, opening IPC memory handle..." << std::endl;
    
    // Open IPC memory handle (using NVIDIA's approach with proper flags)
    unsigned char* d_sharedImageData;
    checkCudaErrors(cudaIpcOpenMemHandle((void**)&d_sharedImageData, header->memHandle, cudaIpcMemLazyEnablePeerAccess));
    
    std::cout << "Spin-Wait IPC Consumer: IPC memory handle opened successfully!" << std::endl;
    std::cout << "  Shared GPU memory address: " << (void*)d_sharedImageData << std::endl;
    
    // Setup CUDA kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Spin-Wait IPC Consumer: Starting MAXIMUM SPEED image processing..." << std::endl;
    
    int lastFrameNumber = -1;
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (true) {
        // PURE SPIN-WAIT for new frame (NO SLEEP!)
        while (!header->dataReady || header->frameNumber == lastFrameNumber) {
            // Busy wait - maximum responsiveness, zero latency
        }
        
        // Process the image directly on shared GPU memory (TRUE ZERO-COPY!)
        processImage<<<gridSize, blockSize>>>(d_sharedImageData, IMAGE_WIDTH, IMAGE_HEIGHT);
        checkCudaErrors(cudaDeviceSynchronize());
        
        // Signal processing complete - NO MEMORY TRANSFERS ANYWHERE!
        header->dataReady = false;
        header->processComplete = true;
        
        lastFrameNumber = header->frameNumber;
        frameCount++;
        
        // FPS calculation
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
        if (duration.count() > 1000) {
            std::cout << "Spin-Wait IPC Consumer processed frame " << lastFrameNumber << " - FPS: " << (frameCount * 1000.0 / duration.count()) << std::endl;
            startTime = currentTime;
            frameCount = 0;
        }
        
        // Optional: Sample pixel values for verification (direct GPU memory read)
        if (lastFrameNumber % 120 == 0) {
            unsigned char samplePixels[3];
            checkCudaErrors(cudaMemcpy(samplePixels, d_sharedImageData, 3, cudaMemcpyDeviceToHost));
            
            std::cout << "Sample processed pixel (0,0): R=" << (int)samplePixels[0] 
                      << " G=" << (int)samplePixels[1] 
                      << " B=" << (int)samplePixels[2] << std::endl;
        }
    }
    
    // Cleanup
    checkCudaErrors(cudaIpcCloseMemHandle(d_sharedImageData));
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    
    std::cout << "Spin-Wait IPC Consumer: Complete!" << std::endl;
    
    return 0;
}