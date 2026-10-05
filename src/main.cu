#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <string>
#include "radix_cpu.h"
#include "radix_cuda.h"
#include "utils.h"

// Number of iterations for benchmarking (can be changed to 5 or 10 for more robustness)
#define NUM_ITERATIONS 3
#define WARMUP_SIZE 1000

void runBenchmark(size_t n, bool is_test = false) {
    std::cout << "Generating random data (size " << n << ")...\n";
    std::vector<uint32_t> data(n);
    generateRandomData(data.data(), n, 42); // fixed seed
    
    std::vector<uint32_t> cpu_data = data;
    std::vector<uint32_t> cuda_data = data;
    
    // CPU timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    radixSortCPU(cpu_data.data(), n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    
    // CUDA timing (average over NUM_ITERATIONS)
    double kernel_time = 0.0;
    double total_time = 0.0;
    
    if (is_test) {
        CudaTimings t = runRadixSortCUDA(cuda_data.data(), n);
        kernel_time = t.kernel_time_ms;
        total_time = t.total_time_ms;
    } else {
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            std::vector<uint32_t> iter_cuda_data = data;
            CudaTimings t = runRadixSortCUDA(iter_cuda_data.data(), n);
            kernel_time += t.kernel_time_ms;
            total_time += t.total_time_ms;
            
            // Only keep the result of the last iteration for validation
            if (i == NUM_ITERATIONS - 1) {
                cuda_data = iter_cuda_data;
            }
        }
        kernel_time /= NUM_ITERATIONS;
        total_time /= NUM_ITERATIONS;
    }
    
    bool passed = validateResults(cpu_data.data(), cuda_data.data(), n);
    
    double k_speedup = cpu_time / kernel_time;
    double e2e_speedup = cpu_time / total_time;
    
    // Print table row
    std::cout << std::left 
              << "| " << std::setw(11) << n 
              << "| " << std::fixed << std::setprecision(2) << std::setw(9) << cpu_time 
              << "| " << std::setw(12) << kernel_time 
              << "| " << std::setw(11) << total_time 
              << "| " << std::setw(12) << k_speedup 
              << "| " << std::setw(14) << e2e_speedup 
              << "|\n";
              
    std::cout << "+------------+------------+-------------+-------------+-------------+----------------+\n\n";
    
    appendToCSV("results/benchmark_results.csv", n, cpu_time, kernel_time, total_time, k_speedup, e2e_speedup, passed);
}

int main(int argc, char** argv) {
    printGPUInformation();
    
    // Determine mode
    bool full_benchmark = false;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--benchmark") {
            full_benchmark = true;
        }
    }
    
    // WARMUP
    std::cout << "Performing CUDA Warm-up (" << WARMUP_SIZE << " elements)...\n";
    std::vector<uint32_t> warmup_data(WARMUP_SIZE);
    generateRandomData(warmup_data.data(), WARMUP_SIZE, 99);
    runRadixSortCUDA(warmup_data.data(), WARMUP_SIZE);
    std::cout << "Warm-up complete.\n\n";

    if (!full_benchmark) {
        std::cout << "Running 1000-element correctness test...\n";
        runBenchmark(1000, true);
        std::cout << "Run with '--benchmark' to execute the full benchmark suite.\n";
        return 0;
    }
    
    // TABLE HEADER
    std::cout << "+------------+------------+-------------+-------------+-------------+----------------+\n";
    std::cout << "| Input Size | CPU (ms)   | Kernel (ms) | Total (ms)  | K-Speedup   | E2E Speedup    |\n";
    std::cout << "+------------+------------+-------------+-------------+-------------+----------------+\n";
    
    size_t sizes[] = {100000, 500000, 1000000, 5000000, 10000000};
    for (size_t s : sizes) {
        runBenchmark(s);
    }
    
    std::cout << "Benchmark results saved to results/benchmark_results.csv\n";
    
    return 0;
}
