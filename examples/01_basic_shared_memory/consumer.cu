#include "../../include/common.h"
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <iostream>

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

struct BasicHeader {
    int frameNumber;
    bool dataReady;
    bool processComplete;
    size_t imageSize;
    bool consumerReady;
};

int main() {
    std::cout << "Basic Shared Memory Consumer: Starting..." << std::endl;
    
    // Initialize CUDA
    cudaSetDevice(0);
    
    // Open shared memory
    HANDLE hMapFile = OpenFileMapping(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        "Local\\BasicSharedMemory"
    );
    
    if (hMapFile == NULL) {
        std::cerr << "Could not open file mapping" << std::endl;
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
    
    // Signal ready
    header->consumerReady = true;
    
    // Allocate GPU memory
    unsigned char* d_imageData;
    cudaMalloc(&d_imageData, IMAGE_SIZE);
    
    // Setup kernel parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Basic Shared Memory Consumer: Waiting for image data..." << std::endl;
    
    int lastFrameNumber = -1;
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (true) {
        // Wait for new frame
        while (!header->dataReady || header->frameNumber == lastFrameNumber) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Copy from shared memory to GPU
        cudaMemcpy(d_imageData, sharedImageData, IMAGE_SIZE, cudaMemcpyHostToDevice);
        
        // Process image
        processImage<<<gridSize, blockSize>>>(d_imageData, IMAGE_WIDTH, IMAGE_HEIGHT);
        cudaDeviceSynchronize();
        
        // Copy back to shared memory
        cudaMemcpy(sharedImageData, d_imageData, IMAGE_SIZE, cudaMemcpyDeviceToHost);
        
        // Signal complete
        header->dataReady = false;
        header->processComplete = true;
        
        lastFrameNumber = header->frameNumber;
        frameCount++;
        
        // FPS calculation
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
        if (duration.count() > 1000) {
            std::cout << "Basic Consumer processed frame " << lastFrameNumber << " - FPS: " << (frameCount * 1000.0 / duration.count()) << std::endl;
            startTime = currentTime;
            frameCount = 0;
        }
    }
    
    // Cleanup
    cudaFree(d_imageData);
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    
    return 0;
}