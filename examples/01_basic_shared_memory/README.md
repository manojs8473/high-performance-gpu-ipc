# Basic Shared Memory Example

Simple baseline implementation using Windows shared memory with standard CPU-GPU transfers.

## Performance
- **~51 FPS** with 4K images
- **1.6 GB/s** data throughput
- **4ms memory transfers** per frame

## How it works
1. Producer generates images on GPU
2. Copies GPU → CPU → Shared Memory
3. Consumer copies Shared Memory → CPU → GPU
4. Processes image on GPU
5. Copies GPU → CPU → Shared Memory

## Trade-offs
- ✅ Simple implementation
- ✅ High compatibility
- ❌ Heavy memory transfer overhead
- ❌ Lower performance

## Build & Run
```bash
# From repository root
mkdir build && cd build
cmake ..
make

# Run (two terminals)
./examples/01_basic_shared_memory/producer
./examples/01_basic_shared_memory/consumer
```

## Key Learning
This example demonstrates the baseline performance and shows why memory transfers are the primary bottleneck in naive implementations.