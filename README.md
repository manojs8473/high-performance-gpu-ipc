# 🚀 High-Performance GPU Inter-Process Communication

[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Performance](https://img.shields.io/badge/Performance-2700+%20FPS-red.svg)](https://github.com/yourusername/gpu-ipc)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**High Performance 4K image processing at 2,700+ FPS using CUDA IPC with zero-copy GPU memory sharing.**

## 🏆 Key Achievements

- **2,700+ FPS** 4K image processing (3840×2160×3 RGB)
- **83.7 GB/s** sustained data throughput
- **True zero-copy** GPU-to-GPU memory sharing
- **0.37ms** frame processing time
- **5,294% improvement** over baseline approaches

## 🎯 Quick Start

```bash
# Clone and build
git clone https://github.com/yourusername/gpu-ipc.git
cd gpu-ipc
mkdir build && cd build
cmake ..
cmake --build . --config Release

# Run highest performance version
./producer_spinwait.exe    # Terminal 1
./consumer_spinwait.exe    # Terminal 2
```

## 📊 Performance Overview

| Implementation | FPS | Throughput | Use Case |
|---------------|-----|------------|----------|
| **🏆 Spin-Wait CUDA IPC** | **2,700+** | **83.7 GB/s** | Maximum performance |
| Sleep-Based CUDA IPC | 280+ | 8.7 GB/s | Production balanced |
| Optimized Shared Memory | 94 | 2.9 GB/s | Fallback compatibility |
| Basic Shared Memory | 51 | 1.6 GB/s | Reference baseline |

## 🔧 System Requirements

- **GPU**: NVIDIA GPU with CUDA Compute Capability 5.0+
- **CUDA**: Version 11.0 or later
- **OS**: Windows 10/11 (Linux support coming soon)
- **Memory**: 4GB+ GPU VRAM for 4K processing
- **Driver**: Latest NVIDIA drivers

## 🚀 Examples

### 1. Highest Performance (Spin-Wait)
```bash
# 2,700+ FPS with maximum CPU usage
./producer_spinwait.exe
./consumer_spinwait.exe
```

### 2. Production Balance (Sleep-Based)
```bash
# 280+ FPS with reasonable CPU usage
./producer_working_ipc.exe
./consumer_working_ipc.exe
```

### 3. Compatibility Fallback
```bash
# 94 FPS using shared memory (no CUDA IPC required)
./producer_sharedmem.exe
./consumer_sharedmem.exe
```

## 📚 Documentation

- **[Research Report](GPU_IPC_Research_Report.md)** - Complete technical analysis
- **[API Reference](docs/API.md)** - Code documentation
- **[Performance Guide](docs/PERFORMANCE.md)** - Optimization techniques
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues

## 🏗️ Architecture

```
Producer Process              Consumer Process
     │                             │
     ├─ GPU Memory ────────────────┤
     │  (CUDA IPC)                 │
     │                             │
     ├─ Image Generation           ├─ Image Processing
     │  generateImage()            │  processImage()
     │                             │
     └─ Spin-Wait Sync ────────────┘
        (Zero Scheduling Overhead)
```

## 🔬 Benchmarking Tools

```bash
# Test your GPU's theoretical limits
./gpu_limits_benchmark.exe

# Analyze memory performance
./memory_benchmark.exe

# Check CUDA IPC compatibility
./cuda_ipc_diagnostic.exe

# Verify with NVIDIA's official sample
./simpleIPC.exe
```

## 🛠️ Build Instructions

### Windows (Visual Studio)
```bash
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### Linux (GCC)
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## 📝 Code Structure

```
gpu-ipc/
├── examples/
│   ├── 01_basic_shared_memory/     # Baseline implementation
│   ├── 02_optimized_shared_memory/ # Improved shared memory
│   ├── 03_cuda_ipc_sleep/          # CUDA IPC with sleep
│   └── 04_cuda_ipc_spinwait/       # Highest performance
├── benchmarks/
│   ├── gpu_limits_benchmark.cu     # GPU theoretical limits
│   ├── memory_benchmark.cu         # Memory performance
│   └── cuda_ipc_diagnostic.cu      # CUDA IPC compatibility
├── include/
│   ├── common.h                    # Shared constants
│   ├── helper_cuda.h               # CUDA utilities
│   └── helper_multiprocess.h       # Multi-process utilities
├── docs/                           # Documentation
└── tests/                          # Unit tests
```

## 🧪 Technical Details

### CUDA IPC Implementation
```cpp
// Producer: Share GPU memory handle
cudaIpcGetMemHandle(&handle, d_gpu_memory);

// Consumer: Open shared GPU memory
cudaIpcOpenMemHandle(&d_shared_memory, handle, cudaIpcMemLazyEnablePeerAccess);

// Process directly on shared GPU memory (zero-copy!)
processImage<<<grid, block>>>(d_shared_memory, width, height);
```

### Spin-Wait Optimization
```cpp
// Eliminated thread scheduling overhead
while (!condition) {
    // Pure spin-wait - zero latency
}
```

## 📈 Performance Analysis

### Bottleneck Evolution
1. **Memory Transfers** (4ms) → Eliminated with CUDA IPC
2. **Thread Scheduling** (3.2ms) → Eliminated with spin-wait
3. **GPU Computation** (0.156ms) → Optimal utilization

### Current Limits
- **GPU Utilization**: 42% of theoretical maximum
- **Remaining Overhead**: 0.214ms IPC synchronization
- **Optimization Potential**: 58% performance gain possible

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for CUDA IPC technology
- CUDA samples community for reference implementations
- Performance optimization insights from GPU computing research

## 📞 Contact

- **Author**: Manoj Sharma
- **Email**: manojs8473@gmail.com
- **LinkedIn**: [LinkedIn](https://www.linkedin.com/in/manoj-sharma-6b59b9155)

## 🔗 Related Projects

- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [GPU Computing Research](https://github.com/gpu-research)
- [High-Performance Computing](https://github.com/hpc-community)

---

⭐ **Star this repository if it helped you achieve high-performance GPU IPC!**