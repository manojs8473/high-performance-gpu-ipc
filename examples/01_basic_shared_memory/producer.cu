#include "../../include/common.h"
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <iostream>

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

struct BasicHeader {
    int frameNumber;
    bool dataReady;
    bool processComplete;
    size_t imageSize;
    bool consumerReady;
};

int main() {
    std::cout << "Basic Shared Memory Producer: Starting..." << std::endl;
    
    // Initialize CUDA
    cudaSetDevice(0);
    
    // Create shared memory
    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(BasicHeader) + IMAGE_SIZE,
        "Local\\BasicSharedMemory"
    );
    
    if (hMapFile == NULL) {
        std::cerr << "Could not create file mapping" << std::endl;
        return 1;
    }
    
    void* pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(BasicHeader) + IMAGE_SIZE);
    if (pBuf == NULL) {
        std::cerr << "Could not map view of file" << std::endl;
        CloseHandle(hMapFile);
        return 1;
    }
    
    BasicHeader* header = (BasicHeader*)pBuf;
    unsigned char* sharedImageData = (unsigned char*)((char*)pBuf + sizeof(BasicHeader));
    
    // Initialize header
    header->frameNumber = 0;
    header->dataReady = false;
    header->processComplete = true;
    header->imageSize = IMAGE_SIZE;
    header->consumerReady = false;
    
    // Allocate GPU memory
    unsigned char* d_imageData;
    cudaMalloc(&d_imageData, IMAGE_SIZE);
    
    // Setup kernel parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    // Wait for consumer
    while (!header->consumerReady) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::cout << "Basic Shared Memory Producer: Starting image generation..." << std::endl;
    
    int frameNumber = 0;
    auto lastTime = std::chrono::high_resolution_clock::now();
    int fpsCounter = 0;
    
    while (true) {
        // Wait for previous frame to be processed
        while (!header->processComplete) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // Target 60 FPS
        }
        
        // Generate image on GPU
        generateImage<<<gridSize, blockSize>>>(d_imageData, frameNumber, IMAGE_WIDTH, IMAGE_HEIGHT);
        cudaDeviceSynchronize();
        
        // Copy to shared memory
        cudaMemcpy(sharedImageData, d_imageData, IMAGE_SIZE, cudaMemcpyDeviceToHost);
        
        // Update header
        header->frameNumber = frameNumber;
        header->processComplete = false;
        header->dataReady = true;
        
        frameNumber++;
        fpsCounter++;
        
        // FPS calculation
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);
        if (duration.count() > 1000) {
            std::cout << "Basic Producer FPS: " << (fpsCounter * 1000.0 / duration.count()) << std::endl;
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