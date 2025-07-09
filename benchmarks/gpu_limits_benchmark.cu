#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

#define IMAGE_SIZE (3840 * 2160 * 3)
#define IMAGE_WIDTH 3840
#define IMAGE_HEIGHT 2160

__global__ void memoryBandwidthTest(unsigned char* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Read and write to test memory bandwidth
    for (int i = idx; i < size; i += stride) {
        data[i] = data[i] + 1;
    }
}

__global__ void generateImageKernel(unsigned char* imageData, int frameNumber, int width, int height) {
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

__global__ void processImageKernel(unsigned char* imageData, int width, int height) {
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

void benchmarkMemoryBandwidth() {
    std::cout << "\n=== GPU Memory Bandwidth Test ===" << std::endl;
    
    unsigned char* d_data;
    checkCudaErrors(cudaMalloc(&d_data, IMAGE_SIZE));
    
    dim3 blockSize(256);
    dim3 gridSize((IMAGE_SIZE + blockSize.x - 1) / blockSize.x);
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        memoryBandwidthTest<<<gridSize, blockSize>>>(d_data, IMAGE_SIZE);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Benchmark
    int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        memoryBandwidthTest<<<gridSize, blockSize>>>(d_data, IMAGE_SIZE);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = duration.count() / (double)iterations / 1000.0; // ms
    double bandwidth = (IMAGE_SIZE * 2 / 1024.0 / 1024.0 / 1024.0) / (avgTime / 1000.0); // GB/s (read + write)
    double maxFPS = 1000.0 / avgTime;
    
    std::cout << "Memory Bandwidth Test Results:" << std::endl;
    std::cout << "  Avg time per 4K frame: " << avgTime << " ms" << std::endl;
    std::cout << "  Memory bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << "  Max FPS (memory bound): " << maxFPS << " FPS" << std::endl;
    
    cudaFree(d_data);
}

void benchmarkImageGeneration() {
    std::cout << "\n=== Image Generation Performance Test ===" << std::endl;
    
    unsigned char* d_imageData;
    checkCudaErrors(cudaMalloc(&d_imageData, IMAGE_SIZE));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        generateImageKernel<<<gridSize, blockSize>>>(d_imageData, i, IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Benchmark
    int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        generateImageKernel<<<gridSize, blockSize>>>(d_imageData, i, IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = duration.count() / (double)iterations / 1000.0; // ms
    double maxFPS = 1000.0 / avgTime;
    
    std::cout << "Image Generation Results:" << std::endl;
    std::cout << "  Avg time per 4K generation: " << avgTime << " ms" << std::endl;
    std::cout << "  Max FPS (generation bound): " << maxFPS << " FPS" << std::endl;
    
    cudaFree(d_imageData);
}

void benchmarkImageProcessing() {
    std::cout << "\n=== Image Processing Performance Test ===" << std::endl;
    
    unsigned char* d_imageData;
    checkCudaErrors(cudaMalloc(&d_imageData, IMAGE_SIZE));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        processImageKernel<<<gridSize, blockSize>>>(d_imageData, IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Benchmark
    int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        processImageKernel<<<gridSize, blockSize>>>(d_imageData, IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = duration.count() / (double)iterations / 1000.0; // ms
    double maxFPS = 1000.0 / avgTime;
    
    std::cout << "Image Processing Results:" << std::endl;
    std::cout << "  Avg time per 4K processing: " << avgTime << " ms" << std::endl;
    std::cout << "  Max FPS (processing bound): " << maxFPS << " FPS" << std::endl;
    
    cudaFree(d_imageData);
}

void benchmarkCombinedWorkload() {
    std::cout << "\n=== Combined Workload Test (Generation + Processing) ===" << std::endl;
    
    unsigned char* d_imageData;
    checkCudaErrors(cudaMalloc(&d_imageData, IMAGE_SIZE));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        generateImageKernel<<<gridSize, blockSize>>>(d_imageData, i, IMAGE_WIDTH, IMAGE_HEIGHT);
        processImageKernel<<<gridSize, blockSize>>>(d_imageData, IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Benchmark
    int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        generateImageKernel<<<gridSize, blockSize>>>(d_imageData, i, IMAGE_WIDTH, IMAGE_HEIGHT);
        processImageKernel<<<gridSize, blockSize>>>(d_imageData, IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = duration.count() / (double)iterations / 1000.0; // ms
    double maxFPS = 1000.0 / avgTime;
    
    std::cout << "Combined Workload Results:" << std::endl;
    std::cout << "  Avg time per frame (gen + proc): " << avgTime << " ms" << std::endl;
    std::cout << "  Max FPS (theoretical limit): " << maxFPS << " FPS" << std::endl;
    
    cudaFree(d_imageData);
}

void benchmarkAsyncOverlap() {
    std::cout << "\n=== Async Stream Overlap Test ===" << std::endl;
    
    unsigned char* d_imageData1;
    unsigned char* d_imageData2;
    checkCudaErrors(cudaMalloc(&d_imageData1, IMAGE_SIZE));
    checkCudaErrors(cudaMalloc(&d_imageData2, IMAGE_SIZE));
    
    cudaStream_t stream1, stream2;
    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaStreamCreate(&stream2));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        generateImageKernel<<<gridSize, blockSize, 0, stream1>>>(d_imageData1, i, IMAGE_WIDTH, IMAGE_HEIGHT);
        processImageKernel<<<gridSize, blockSize, 0, stream2>>>(d_imageData2, IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Benchmark overlapped execution
    int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        generateImageKernel<<<gridSize, blockSize, 0, stream1>>>(d_imageData1, i, IMAGE_WIDTH, IMAGE_HEIGHT);
        processImageKernel<<<gridSize, blockSize, 0, stream2>>>(d_imageData2, IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = duration.count() / (double)iterations / 1000.0; // ms
    double maxFPS = 1000.0 / avgTime;
    
    std::cout << "Async Overlap Results:" << std::endl;
    std::cout << "  Avg time per frame (overlapped): " << avgTime << " ms" << std::endl;
    std::cout << "  Max FPS (with overlap): " << maxFPS << " FPS" << std::endl;
    
    checkCudaErrors(cudaStreamDestroy(stream1));
    checkCudaErrors(cudaStreamDestroy(stream2));
    cudaFree(d_imageData1);
    cudaFree(d_imageData2);
}

int main() {
    std::cout << "=== RTX 3080 Ti Theoretical Performance Limits ===" << std::endl;
    std::cout << "Testing 4K RGB images (31MB per frame)" << std::endl;
    
    // Get GPU specs
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    std::cout << "\nGPU: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Memory Clock: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    
    // Calculate theoretical memory bandwidth
    double theoreticalBandwidth = (prop.memoryClockRate * 2.0 * prop.memoryBusWidth / 8.0) / 1000000.0; // GB/s
    std::cout << "Theoretical Memory Bandwidth: " << theoreticalBandwidth << " GB/s" << std::endl;
    
    // Run benchmarks
    benchmarkMemoryBandwidth();
    benchmarkImageGeneration();
    benchmarkImageProcessing();
    benchmarkCombinedWorkload();
    benchmarkAsyncOverlap();
    
    std::cout << "\n=== Analysis ===" << std::endl;
    std::cout << "Your achieved 280 FPS compares to theoretical limits above." << std::endl;
    std::cout << "If 280 FPS is close to the combined workload result," << std::endl;
    std::cout << "then you're near the GPU's theoretical maximum!" << std::endl;
    
    return 0;
}