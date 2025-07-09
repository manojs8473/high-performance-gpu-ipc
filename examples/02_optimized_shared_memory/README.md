# Optimized Shared Memory Example

Enhanced shared memory implementation with pinned memory and async operations.

## Performance
- **~94 FPS** with 4K images
- **2.9 GB/s** data throughput
- **4ms memory transfers** per frame (optimized)

## Key Optimizations
1. **Pinned Memory**: `cudaHostAlloc()` for faster transfers
2. **CUDA Streams**: Async operations for overlapping
3. **Atomic Synchronization**: `std::atomic` for coordination
4. **Reduced Polling**: Microsecond-level synchronization

## How it works
1. Producer generates images on GPU
2. Async copy GPU → Pinned Memory → Shared Memory
3. Consumer copies Shared Memory → Pinned Memory → GPU
4. Processes image on GPU with async operations
5. Async copy GPU → Pinned Memory → Shared Memory

## Trade-offs
- ✅ Significantly better performance
- ✅ Good compatibility
- ✅ Lower CPU overhead
- ❌ Still limited by PCIe bandwidth

## Build & Run
```bash
# From repository root
mkdir build && cd build
cmake ..
make

# Run (two terminals)
./examples/02_optimized_shared_memory/producer
./examples/02_optimized_shared_memory/consumer
```

## Key Learning
This example shows how memory optimization can nearly double performance, but PCIe bandwidth remains the fundamental limit.