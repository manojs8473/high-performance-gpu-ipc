#include "common.h"
#include <cuda_runtime.h>
#include <chrono>
#include <thread>

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

struct SharedMemoryHeader {
    int frameNumber;
    bool dataReady;
    bool processComplete;
    size_t imageSize;
};

int main() {
    std::cout << "Producer: Starting..." << std::endl;
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // Create shared memory for image data
    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(SharedMemoryHeader) + IMAGE_SIZE,
"Local\\CudaImageSharedMemory"
    );
    
    if (hMapFile == NULL) {
        std::cerr << "Could not create file mapping object: " << GetLastError() << std::endl;
        return 1;
    }
    
    void* pBuf = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(SharedMemoryHeader) + IMAGE_SIZE
    );
    
    if (pBuf == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return 1;
    }
    
    SharedMemoryHeader* header = (SharedMemoryHeader*)pBuf;
    unsigned char* sharedImageData = (unsigned char*)((char*)pBuf + sizeof(SharedMemoryHeader));
    
    // Initialize header
    header->frameNumber = 0;
    header->dataReady = false;
    header->processComplete = false;
    header->imageSize = IMAGE_SIZE;
    
    // Allocate GPU memory for image generation
    unsigned char* d_imageData;
    CUDA_CHECK(cudaMalloc(&d_imageData, IMAGE_SIZE));
    
    // Setup CUDA kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Producer: Waiting for consumer..." << std::endl;
    
    // Wait for consumer to connect
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "Producer: Starting image generation..." << std::endl;
    
    int frameNumber = 0;
    auto lastTime = std::chrono::high_resolution_clock::now();
    int fpsCounter = 0;
    
    while (true) {
        // Wait for previous frame to be processed
        while (header->dataReady && !header->processComplete) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        // Generate image on GPU
        generateImage<<<gridSize, blockSize>>>(d_imageData, frameNumber, IMAGE_WIDTH, IMAGE_HEIGHT);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy from GPU to shared memory
        CUDA_CHECK(cudaMemcpy(sharedImageData, d_imageData, IMAGE_SIZE, cudaMemcpyDeviceToHost));
        
        // Update header
        header->frameNumber = frameNumber;
        header->dataReady = true;
        header->processComplete = false;
        
        frameNumber++;
        fpsCounter++;
        
        // FPS calculation
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);
        if (duration.count() > 1000) {
            std::cout << "Producer FPS: " << (fpsCounter * 1000.0 / duration.count()) << std::endl;
            lastTime = currentTime;
            fpsCounter = 0;
        }
        
        // No sleep - run at maximum speed
        // std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    
    // Cleanup
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    cudaFree(d_imageData);
    
    return 0;
}