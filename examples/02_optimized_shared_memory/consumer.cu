#include "common.h"
#include <cuda_runtime.h>
#include <chrono>
#include <thread>

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

struct SharedMemoryHeader {
    int frameNumber;
    bool dataReady;
    bool processComplete;
    size_t imageSize;
};

int main() {
    std::cout << "Consumer: Starting..." << std::endl;
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // Open shared memory
    HANDLE hMapFile = OpenFileMapping(
        FILE_MAP_ALL_ACCESS,
        FALSE,
"Local\\CudaImageSharedMemory"
    );
    
    if (hMapFile == NULL) {
        std::cerr << "Could not open file mapping object: " << GetLastError() << std::endl;
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
    
    // Signal that consumer is ready
    header->dataReady = true;
    
    // Allocate GPU memory for processing
    unsigned char* d_imageData;
    CUDA_CHECK(cudaMalloc(&d_imageData, IMAGE_SIZE));
    
    // Setup CUDA kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Consumer: Connected, waiting for image data..." << std::endl;
    
    int lastFrameNumber = -1;
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (true) {
        // Wait for new frame (optimized polling)
        while (!header->dataReady || header->frameNumber == lastFrameNumber) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        // Copy from shared memory to GPU
        CUDA_CHECK(cudaMemcpy(d_imageData, sharedImageData, IMAGE_SIZE, cudaMemcpyHostToDevice));
        
        // Process the image on GPU
        processImage<<<gridSize, blockSize>>>(d_imageData, IMAGE_WIDTH, IMAGE_HEIGHT);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy processed image back to shared memory
        CUDA_CHECK(cudaMemcpy(sharedImageData, d_imageData, IMAGE_SIZE, cudaMemcpyDeviceToHost));
        
        // Signal processing complete
        header->processComplete = true;
        lastFrameNumber = header->frameNumber;
        frameCount++;
        
        // FPS calculation
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
        if (duration.count() > 1000) {
            std::cout << "Consumer processed frame " << header->frameNumber << " - FPS: " << (frameCount * 1000.0 / duration.count()) << std::endl;
            startTime = currentTime;
            frameCount = 0;
        }
        
        // Optional: Sample pixel values for verification
        if (header->frameNumber % 60 == 0) {
            unsigned char* h_imageData = new unsigned char[IMAGE_SIZE];
            CUDA_CHECK(cudaMemcpy(h_imageData, d_imageData, IMAGE_SIZE, cudaMemcpyDeviceToHost));
            
            std::cout << "Sample processed pixel (0,0): R=" << (int)h_imageData[0] 
                      << " G=" << (int)h_imageData[1] 
                      << " B=" << (int)h_imageData[2] << std::endl;
            
            delete[] h_imageData;
        }
    }
    
    // Cleanup
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    cudaFree(d_imageData);
    
    return 0;
}