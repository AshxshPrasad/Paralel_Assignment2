#ifndef RADIX_CUDA_H
#define RADIX_CUDA_H

#include <cstdint>
#include <cstddef>

// Helper to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Performs a 1-bit per pass Radix Sort (32 passes) on the GPU.
// Returns the kernel execution time in milliseconds.
// Total execution time (including memory transfers) should be measured externally.
double radixSortCUDA(uint32_t* d_data, size_t n);

// Wrapper to allocate memory, run the sort, and copy back, and measure total time.
// Returns a pair of {kernel_time_ms, total_time_ms}.
struct CudaTimings {
    double kernel_time_ms;
    double total_time_ms;
};

CudaTimings runRadixSortCUDA(uint32_t* h_data, size_t n);

#endif // RADIX_CUDA_H
