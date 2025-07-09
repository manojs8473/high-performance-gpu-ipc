#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define IMAGE_SIZE (3840 * 2160 * 3)

__global__ void testKernel(unsigned char* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] + 1) % 256;
    }
}

void benchmarkMemoryType(const char* name, unsigned char* d_data, int iterations) {
    dim3 blockSize(256);
    dim3 gridSize((IMAGE_SIZE + blockSize.x - 1) / blockSize.x);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        testKernel<<<gridSize, blockSize>>>(d_data, IMAGE_SIZE);
        cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = duration.count() / (double)iterations / 1000.0; // ms
    double bandwidth = (IMAGE_SIZE / 1024.0 / 1024.0) / (avgTime / 1000.0); // MB/s
    
    std::cout << name << ":" << std::endl;
    std::cout << "  Avg time per operation: " << avgTime << " ms" << std::endl;
    std::cout << "  Bandwidth: " << bandwidth << " MB/s" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "=== GPU Memory Performance Benchmark ===" << std::endl;
    std::cout << "Image size: " << IMAGE_SIZE / (1024*1024) << " MB" << std::endl;
    std::cout << std::endl;
    
    // Test 1: Regular GPU memory
    unsigned char* d_gpu_mem;
    cudaMalloc(&d_gpu_mem, IMAGE_SIZE);
    benchmarkMemoryType("Regular GPU Memory", d_gpu_mem, 100);
    
    // Test 2: Mapped host memory
    unsigned char* h_mapped_mem;
    cudaHostAlloc(&h_mapped_mem, IMAGE_SIZE, cudaHostAllocMapped);
    
    unsigned char* d_mapped_mem;
    cudaHostGetDevicePointer(&d_mapped_mem, h_mapped_mem, 0);
    benchmarkMemoryType("Mapped Host Memory", d_mapped_mem, 100);
    
    // Test 3: Pinned memory transfer speed
    unsigned char* h_pinned_mem;
    cudaHostAlloc(&h_pinned_mem, IMAGE_SIZE, cudaHostAllocDefault);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        cudaMemcpy(h_pinned_mem, d_gpu_mem, IMAGE_SIZE, cudaMemcpyDeviceToHost);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTransfer = duration.count() / 100.0 / 1000.0; // ms
    double transferBandwidth = (IMAGE_SIZE / 1024.0 / 1024.0) / (avgTransfer / 1000.0); // MB/s
    
    std::cout << "Pinned Memory Transfer (D2H):" << std::endl;
    std::cout << "  Avg transfer time: " << avgTransfer << " ms" << std::endl;
    std::cout << "  Transfer bandwidth: " << transferBandwidth << " MB/s" << std::endl;
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_gpu_mem);
    cudaFreeHost(h_mapped_mem);
    cudaFreeHost(h_pinned_mem);
    
    std::cout << "Conclusion: If mapped memory is much slower," << std::endl;
    std::cout << "then pinned memory + fast transfers is better." << std::endl;
    
    return 0;
}