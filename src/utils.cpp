#include "utils.h"
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

void generateRandomData(uint32_t* data, size_t n, unsigned int seed) {
    std::mt19937 eng(seed);

    // Use full range of uint32_t
    std::uniform_int_distribution<uint32_t> dist(
        0, std::numeric_limits<uint32_t>::max()
    );

    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(eng);
    }
}

bool validateResults(const uint32_t* cpu_data, const uint32_t* cuda_data, size_t n) {
    bool passed = true;
    
    std::cout << "Correctness Verification\n";
    std::cout << "------------------------\n";
    std::cout << "Input Size: " << n << "\n";
    
    // Check match
    bool match = true;
    for (size_t i = 0; i < n; ++i) {
        if (cpu_data[i] != cuda_data[i]) {
            std::cout << "Mismatch at index " << i << "!\n";
            std::cout << "CPU value:  " << cpu_data[i] << "\n";
            std::cout << "CUDA value: " << cuda_data[i] << "\n";
            match = false;
            passed = false;
            break;
        }
    }
    
    if (match) {
        std::cout << "CPU vs CUDA: PASS\n";
    } else {
        std::cout << "CPU vs CUDA: FAIL\n";
    }
    
    // Check if sorted
    bool sorted = true;
    for (size_t i = 1; i < n; ++i) {
        if (cpu_data[i - 1] > cpu_data[i]) {
            sorted = false;
            passed = false;
            break;
        }
    }
    
    if (sorted) {
        std::cout << "Sorted Check: PASS\n";
    } else {
        std::cout << "Sorted Check: FAIL\n";
    }
    
    return passed;
}

void printGPUInformation() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in cudaGetDeviceCount: " << cudaGetErrorString(error) << "\n";
        return;
    }
    
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found.\n";
        return;
    }
    
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    
    std::cout << "==================================================\n";
    std::cout << "CUDA GPU Information\n";
    std::cout << "==================================================\n";
    std::cout << "CUDA Driver Version / Runtime Version: " 
              << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << " / "
              << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << "\n";
              
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "Device " << i << ": \"" << deviceProp.name << "\"\n";
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Multiprocessors (SMs): " << deviceProp.multiProcessorCount << "\n";
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << "\n";
    }
    std::cout << "==================================================\n\n";
}

void appendToCSV(const std::string& filename, size_t input_size, double cpu_time, double kernel_time, double total_time, double kernel_speedup, double e2e_speedup, bool correctness) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        // Write header if file is empty
        file.seekp(0, std::ios::end);
        if (file.tellp() == 0) {
            file << "input_size,cpu_time_ms,cuda_kernel_time_ms,cuda_total_time_ms,kernel_speedup,end_to_end_speedup,correctness\n";
        }
        
        file << input_size << ","
             << std::fixed << std::setprecision(3) << cpu_time << ","
             << kernel_time << ","
             << total_time << ","
             << kernel_speedup << ","
             << e2e_speedup << ","
             << (correctness ? "PASS" : "FAIL") << "\n";
        file.close();
    } else {
        std::cerr << "Could not open file " << filename << " for writing.\n";
    }
}
