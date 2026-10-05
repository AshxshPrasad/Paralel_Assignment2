#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <cstddef>
#include <string>

// Generates an array of random 32-bit unsigned integers.
void generateRandomData(uint32_t* data, size_t n, unsigned int seed);

// Validates the CUDA array against the CPU array and checks if they are sorted.
// Returns true if PASS, false if FAIL.
bool validateResults(const uint32_t* cpu_data, const uint32_t* cuda_data, size_t n);

// Prints a summary of the available CUDA devices
void printGPUInformation();

// Write results to CSV
void appendToCSV(const std::string& filename, size_t input_size, double cpu_time, double kernel_time, double total_time, double kernel_speedup, double e2e_speedup, bool correctness);

#endif // UTILS_H
