#include <cuda_runtime.h>
#include <iostream>
#include <string>

void checkCudaError(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) {
        std::cout << "❌ " << operation << " failed: " << cudaGetErrorString(error) << " (code: " << error << ")" << std::endl;
    } else {
        std::cout << "✅ " << operation << " succeeded" << std::endl;
    }
}

int main() {
    std::cout << "=== CUDA IPC Diagnostic Tool ===" << std::endl;
    
    // 1. Check CUDA version and driver
    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    
    std::cout << "\n1. CUDA Version Information:" << std::endl;
    std::cout << "   Driver Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;
    std::cout << "   Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
    
    // 2. Check GPU properties
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "\n2. GPU Information:" << std::endl;
    std::cout << "   Device Count: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "   Device " << i << ": " << prop.name << std::endl;
        std::cout << "     Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "     Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "     Unified Addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << "     TCC Driver Mode: " << (prop.tccDriver ? "Yes" : "No") << std::endl;
        
        // Check if GPU supports IPC
        int canAccessPeer;
        if (deviceCount > 1) {
            cudaDeviceCanAccessPeer(&canAccessPeer, i, (i + 1) % deviceCount);
            std::cout << "     Can Access Peer: " << (canAccessPeer ? "Yes" : "No") << std::endl;
        }
    }
    
    // 3. Test basic CUDA operations
    std::cout << "\n3. Basic CUDA Operations:" << std::endl;
    
    cudaError_t error = cudaSetDevice(0);
    checkCudaError(error, "cudaSetDevice(0)");
    
    // 4. Test memory allocation
    std::cout << "\n4. Memory Allocation Tests:" << std::endl;
    
    void* d_ptr;
    error = cudaMalloc(&d_ptr, 1024 * 1024); // 1MB
    checkCudaError(error, "cudaMalloc(1MB)");
    
    if (error == cudaSuccess) {
        // 5. Test IPC handle creation
        std::cout << "\n5. IPC Handle Creation Test:" << std::endl;
        
        cudaIpcMemHandle_t handle;
        error = cudaIpcGetMemHandle(&handle, d_ptr);
        checkCudaError(error, "cudaIpcGetMemHandle");
        
        if (error == cudaSuccess) {
            std::cout << "   IPC Handle created successfully!" << std::endl;
            std::cout << "   Handle bytes: ";
            for (int i = 0; i < 64; i++) {
                printf("%02x ", ((unsigned char*)&handle)[i]);
            }
            std::cout << std::endl;
            
            // 6. Test IPC handle opening (same process)
            std::cout << "\n6. IPC Handle Opening Test (same process):" << std::endl;
            
            void* d_ptr2;
            error = cudaIpcOpenMemHandle(&d_ptr2, handle, cudaIpcMemLazyEnablePeerAccess);
            checkCudaError(error, "cudaIpcOpenMemHandle (same process)");
            
            if (error == cudaSuccess) {
                std::cout << "   Original pointer: " << d_ptr << std::endl;
                std::cout << "   IPC pointer: " << d_ptr2 << std::endl;
                
                // Test memory access
                int testValue = 42;
                error = cudaMemcpy(d_ptr, &testValue, sizeof(int), cudaMemcpyHostToDevice);
                checkCudaError(error, "cudaMemcpy to original pointer");
                
                int readValue;
                error = cudaMemcpy(&readValue, d_ptr2, sizeof(int), cudaMemcpyDeviceToHost);
                checkCudaError(error, "cudaMemcpy from IPC pointer");
                
                if (readValue == testValue) {
                    std::cout << "   ✅ Memory access through IPC handle works!" << std::endl;
                } else {
                    std::cout << "   ❌ Memory access through IPC handle failed!" << std::endl;
                }
                
                cudaIpcCloseMemHandle(d_ptr2);
            }
        } else {
            std::cout << "\n   Common reasons for IPC failure:" << std::endl;
            std::cout << "   - GPU doesn't support IPC (older cards)" << std::endl;
            std::cout << "   - Running on integrated GPU (Intel/AMD)" << std::endl;
            std::cout << "   - Windows display driver vs TCC driver" << std::endl;
            std::cout << "   - Virtualized environment" << std::endl;
            std::cout << "   - WSL/Docker limitations" << std::endl;
        }
        
        cudaFree(d_ptr);
    }
    
    // 7. Check for specific error conditions
    std::cout << "\n7. Environment Checks:" << std::endl;
    
    // Check if running in WSL
    std::cout << "   WSL Environment: ";
    if (system("uname -r | grep -q microsoft") == 0) {
        std::cout << "Yes (may limit IPC)" << std::endl;
    } else {
        std::cout << "No" << std::endl;
    }
    
    // Check driver mode
    std::cout << "   Driver Mode: ";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.tccDriver) {
        std::cout << "TCC (better for IPC)" << std::endl;
    } else {
        std::cout << "WDDM (may limit IPC)" << std::endl;
    }
    
    std::cout << "\n=== Diagnostic Complete ===" << std::endl;
    
    return 0;
}