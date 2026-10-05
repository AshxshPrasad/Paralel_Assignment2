#include "radix_cuda.h"
#include <cstdio>
#include <cuda_runtime.h>

// ---------------------------------------------------------
// CUDA KERNELS
// ---------------------------------------------------------

// 1. Predicate kernel: writes 1 if the bit is 0, else 0
__global__ void predicateKernel(uint32_t* d_predicate, const uint32_t* d_data, int bit, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t bit_val = (d_data[idx] >> bit) & 1;
        d_predicate[idx] = (bit_val == 0) ? 1 : 0;
    }
}

// 2. Local exclusive scan kernel using Hillis-Steele
__global__ void scanLocalBlocks(uint32_t* d_out, const uint32_t* d_in, uint32_t* d_block_sums, int n) {
    extern __shared__ uint32_t temp[]; // Size: 2 * blockDim.x
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    int pout = 0;
    int pin = 1;
    
    // Load input into shared memory
    temp[pout * blockDim.x + tid] = (idx < n) ? d_in[idx] : 0;
    __syncthreads();
    
    // Hillis-Steele inclusive scan
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;
        
        if (tid >= offset) {
            temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid] + temp[pin * blockDim.x + tid - offset];
        } else {
            temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid];
        }
        __syncthreads();
    }
    
    // Read the inclusive scan result
    uint32_t val = temp[pout * blockDim.x + tid];
    
    // Write the total sum of the block to d_block_sums
    if (tid == blockDim.x - 1) {
        if (d_block_sums) {
            d_block_sums[blockIdx.x] = val;
        }
    }
    
    // Shift right to make it an exclusive scan
    uint32_t exc_val = (tid > 0) ? temp[pout * blockDim.x + tid - 1] : 0;
    
    // Write exclusive scan result to output
    if (idx < n) {
        d_out[idx] = exc_val;
    }
}

// 3. Add block sums kernel
__global__ void addBlockSums(uint32_t* d_out, const uint32_t* d_block_sums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && blockIdx.x > 0) {
        // Add the scanned block sum to the local exclusive scan
        d_out[idx] += d_block_sums[blockIdx.x];
    }
}

// 4. Compute total zeros kernel (single thread)
__global__ void computeTotalZeros(uint32_t* d_total_zeros, const uint32_t* d_predicate, const uint32_t* d_scan, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_total_zeros[0] = d_scan[n - 1] + d_predicate[n - 1];
    }
}

// 5. Scatter kernel
__global__ void scatterKernel(uint32_t* d_out_data, const uint32_t* d_in_data, const uint32_t* d_predicate, const uint32_t* d_scan, const uint32_t* d_total_zeros, int bit, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t val = d_in_data[idx];
        uint32_t bit_val = (val >> bit) & 1;
        uint32_t dest_idx;
        
        if (bit_val == 0) {
            dest_idx = d_scan[idx];
        } else {
            dest_idx = d_total_zeros[0] + (idx - d_scan[idx]);
        }
        
        d_out_data[dest_idx] = val;
    }
}

// ---------------------------------------------------------
// HOST ORCHESTRATION
// ---------------------------------------------------------

void recursiveScan(uint32_t* d_out, uint32_t* d_in, uint32_t* d_block_sums_workspace, int n, int block_size) {
    int num_blocks = (n + block_size - 1) / block_size;
    size_t shared_mem_size = 2 * block_size * sizeof(uint32_t);
    
    if (num_blocks <= 1) {
        scanLocalBlocks<<<1, block_size, shared_mem_size>>>(d_out, d_in, nullptr, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        scanLocalBlocks<<<num_blocks, block_size, shared_mem_size>>>(d_out, d_in, d_block_sums_workspace, n);
        CUDA_CHECK(cudaGetLastError());
        
        // Recursively scan the block sums
        // Shift workspace pointer to avoid overwriting current level's block sums
        uint32_t* next_workspace = d_block_sums_workspace + num_blocks;
        recursiveScan(d_block_sums_workspace, d_block_sums_workspace, next_workspace, num_blocks, block_size);
        
        // Add scanned block sums back to local scans
        addBlockSums<<<num_blocks, block_size>>>(d_out, d_block_sums_workspace, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

double radixSortCUDA(uint32_t* d_data, size_t n) {
    if (n == 0) return 0.0;

    int block_size = 512;
    int num_blocks = (n + block_size - 1) / block_size;
    
    // Allocate device workspace
    uint32_t *d_temp_data, *d_predicate, *d_scan, *d_block_sums, *d_total_zeros;
    CUDA_CHECK(cudaMalloc(&d_temp_data, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_predicate, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_scan, n * sizeof(uint32_t)));
    
    // block_sums needs to handle multiple levels of recursion
    // N elements -> N/512 blocks -> N/512/512 blocks ...
    // Total size is bounded by N/512 + N/256K + ... < N/256
    size_t block_sums_size = (n / block_size) + block_size; // Safe upper bound
    CUDA_CHECK(cudaMalloc(&d_block_sums, block_sums_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_total_zeros, sizeof(uint32_t)));
    
    uint32_t* d_src = d_data;
    uint32_t* d_dst = d_temp_data;
    
    // Setup CUDA events for kernel timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int bit = 0; bit < 32; ++bit) {
        // 1. Predicate Generation
        predicateKernel<<<num_blocks, block_size>>>(d_predicate, d_src, bit, n);
        CUDA_CHECK(cudaGetLastError());
        
        // 2. Prefix Scan (Exclusive)
        recursiveScan(d_scan, d_predicate, d_block_sums, n, block_size);
        
        // 3. Compute Total Zeros
        computeTotalZeros<<<1, 1>>>(d_total_zeros, d_predicate, d_scan, n);
        CUDA_CHECK(cudaGetLastError());
        
        // 4. Scatter
        scatterKernel<<<num_blocks, block_size>>>(d_dst, d_src, d_predicate, d_scan, d_total_zeros, bit, n);
        CUDA_CHECK(cudaGetLastError());
        
        // 5. Swap pointers
        uint32_t* temp_ptr = d_src;
        d_src = d_dst;
        d_dst = temp_ptr;
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernel_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));
    
    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_temp_data));
    CUDA_CHECK(cudaFree(d_predicate));
    CUDA_CHECK(cudaFree(d_scan));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_total_zeros));
    
    return static_cast<double>(kernel_time_ms);
}

CudaTimings runRadixSortCUDA(uint32_t* h_data, size_t n) {
    CudaTimings timings = {0.0, 0.0};
    if (n == 0) return timings;
    
    // Total time includes H2D and D2H transfers
    cudaEvent_t start_total, stop_total;
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    
    CUDA_CHECK(cudaEventRecord(start_total));
    
    uint32_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    // Run the radix sort and get kernel time
    timings.kernel_time_ms = radixSortCUDA(d_data, n);
    
    // Result is in d_data (because 32 passes implies an even number of swaps)
    CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));
    
    float total_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, start_total, stop_total));
    timings.total_time_ms = total_time_ms;
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    
    return timings;
}
