# Troubleshooting Guide

## Common Issues and Solutions

### CUDA IPC Failures

#### "invalid device context" (Error 201)
**Symptoms:**
```
cudaIpcOpenMemHandle failed: invalid device context
```

**Solutions:**
1. **Check GPU compatibility:**
   ```bash
   ./cuda_ipc_diagnostic.exe
   ```

2. **Verify unified addressing:**
   ```cpp
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, 0);
   if (!prop.unifiedAddressing) {
       // GPU doesn't support CUDA IPC
   }
   ```

3. **Check compute mode:**
   ```cpp
   if (prop.computeMode != cudaComputeModeDefault) {
       // GPU in wrong compute mode
   }
   ```

#### "invalid argument" (Error 11)
**Symptoms:**
```
cudaIpcGetMemHandle failed: invalid argument
```

**Solutions:**
1. **Use proper memory allocation:**
   ```cpp
   // ✅ Good: Standard allocation
   cudaMalloc(&d_ptr, size);
   
   // ❌ Bad: Mapped memory
   cudaHostAlloc(&h_ptr, size, cudaHostAllocMapped);
   ```

2. **Check memory alignment:**
   ```cpp
   // Ensure memory is properly aligned
   size_t alignedSize = ((size + 255) / 256) * 256;
   cudaMalloc(&d_ptr, alignedSize);
   ```

### Performance Issues

#### Low FPS Performance
**Symptoms:**
- FPS significantly below expected values
- High CPU usage
- Long frame processing times

**Solutions:**
1. **Profile with benchmarks:**
   ```bash
   ./gpu_limits_benchmark.exe
   ./memory_benchmark.exe
   ```

2. **Check thread scheduling:**
   ```cpp
   // Replace sleep with spin-wait for maximum performance
   while (!condition) {
       // Pure spin-wait instead of sleep
   }
   ```

3. **Optimize memory transfers:**
   ```cpp
   // Use pinned memory
   cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault);
   
   // Use async operations
   cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
   ```

#### Memory Bandwidth Bottlenecks
**Symptoms:**
- Low throughput despite high FPS
- Memory-bound performance

**Solutions:**
1. **Switch to CUDA IPC:**
   ```cpp
   // Eliminate CPU-GPU transfers entirely
   cudaIpcGetMemHandle(&handle, d_gpu_memory);
   ```

2. **Use memory coalescing:**
   ```cpp
   // Ensure coalesced access patterns
   int pixelIdx = idy * width + idx; // Sequential access
   ```

### Build Issues

#### Missing CUDA Headers
**Symptoms:**
```
fatal error: cuda_runtime.h: No such file or directory
```

**Solutions:**
1. **Install CUDA Toolkit:**
   - Download from NVIDIA website
   - Ensure PATH includes CUDA binaries

2. **Check CMake configuration:**
   ```cmake
   find_package(CUDA REQUIRED)
   enable_language(CUDA)
   ```

#### Architecture Compatibility
**Symptoms:**
```
nvcc fatal: Unsupported gpu architecture 'compute_89'
```

**Solutions:**
1. **Update CMakeLists.txt:**
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86)
   ```

2. **Check your GPU's compute capability:**
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```

### Runtime Issues

#### Shared Memory Access Errors
**Symptoms:**
```
Could not create file mapping object: 5
Could not open file mapping object: 2
```

**Solutions:**
1. **Run as administrator** (if required)
2. **Check Windows permissions**
3. **Verify shared memory names are unique**

#### Process Synchronization Issues
**Symptoms:**
- Processes hang waiting for each other
- Inconsistent frame processing

**Solutions:**
1. **Check initialization order:**
   ```cpp
   // Producer should create shared memory first
   // Consumer should connect after producer is ready
   ```

2. **Add timeout mechanisms:**
   ```cpp
   auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
   while (!condition && std::chrono::steady_clock::now() < timeout) {
       std::this_thread::sleep_for(std::chrono::milliseconds(10));
   }
   ```

## Debugging Tools

### CUDA Debugging
```bash
# Run with CUDA debugging
set CUDA_LAUNCH_BLOCKING=1
./your_program.exe

# Check CUDA errors
cuda-memcheck ./your_program.exe
```

### Performance Profiling
```bash
# Profile GPU usage
nsight-sys profile ./your_program.exe

# Profile memory usage
nsight-compute ./your_program.exe
```

### System Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
perfmon.exe
```

## System Requirements Verification

### GPU Requirements
- **Compute Capability**: 5.0 or higher
- **Memory**: 4GB+ for 4K processing
- **Driver**: Latest NVIDIA drivers

### Software Requirements
- **CUDA Toolkit**: 11.0+
- **Visual Studio**: 2019 or later (Windows)
- **CMake**: 3.18+

### Performance Expectations

| GPU Generation | Expected FPS (4K) | Notes |
|---------------|-------------------|-------|
| RTX 40xx | 3,000+ | Latest architecture |
| RTX 30xx | 2,500+ | High-end performance |
| RTX 20xx | 1,500+ | Good performance |
| GTX 16xx | 800+ | Entry-level |
| GTX 10xx | 400+ | Older generation |

## Getting Help

### Before Reporting Issues
1. Run diagnostic tools
2. Check system requirements
3. Verify CUDA installation
4. Test with simple examples first

### Information to Include
- GPU model and driver version
- CUDA version
- Operating system
- Complete error messages
- Performance measurements
- System specifications

### Resources
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [GitHub Issues](https://github.com/yourusername/gpu-ipc/issues)