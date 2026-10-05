#ifndef RADIX_CPU_H
#define RADIX_CPU_H

#include <cstdint>
#include <cstddef>

// Performs a 1-bit per pass Radix Sort (32 passes) on the CPU.
// Does not use any standard library sorting functions.
void radixSortCPU(uint32_t* data, size_t n);

#endif // RADIX_CPU_H
