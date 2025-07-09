# API Reference

## Core Functions

### Image Generation
```cpp
__global__ void generateImage(unsigned char* imageData, int frameNumber, int width, int height)
```
Generates animated 4K RGB images on GPU.

**Parameters:**
- `imageData`: GPU memory pointer for image data
- `frameNumber`: Current frame number for animation
- `width`: Image width (default: 3840)
- `height`: Image height (default: 2160)

### Image Processing
```cpp
__global__ void processImage(unsigned char* imageData, int width, int height)
```
Processes images on GPU (color inversion example).

**Parameters:**
- `imageData`: GPU memory pointer for image data
- `width`: Image width
- `height`: Image height

## CUDA IPC Functions

### Memory Handle Creation
```cpp
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr)
```
Creates IPC handle for GPU memory sharing.

### Memory Handle Opening
```cpp
cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
```
Opens shared GPU memory in another process.

**Recommended flags:**
- `cudaIpcMemLazyEnablePeerAccess`: Enable peer access as needed

### Memory Handle Closing
```cpp
cudaError_t cudaIpcCloseMemHandle(void* devPtr)
```
Closes shared GPU memory handle.

## Synchronization Structures

### Basic Header
```cpp
struct BasicHeader {
    int frameNumber;
    bool dataReady;
    bool processComplete;
    size_t imageSize;
    bool consumerReady;
};
```

### Spin-Wait Header
```cpp
struct IPCHeader {
    cudaIpcMemHandle_t memHandle;
    volatile int frameNumber;
    volatile bool dataReady;
    volatile bool processComplete;
    size_t imageSize;
    volatile bool consumerReady;
};
```

## Constants

```cpp
#define IMAGE_WIDTH 3840
#define IMAGE_HEIGHT 2160
#define IMAGE_CHANNELS 3
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS)
```

## Error Handling

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)
```

## Performance Monitoring

### FPS Calculation
```cpp
auto currentTime = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);
if (duration.count() > 1000) {
    double fps = (frameCount * 1000.0 / duration.count());
    std::cout << "FPS: " << fps << std::endl;
}
```

### Throughput Calculation
```cpp
double throughput = (fps * IMAGE_SIZE) / (1024.0 * 1024.0 * 1024.0); // GB/s
```