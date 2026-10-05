#include "radix_cpu.h"
#include <vector>

void radixSortCPU(uint32_t* data, size_t n) {
    if (n == 0) return;

    // We use a secondary buffer for scattering elements
    std::vector<uint32_t> temp(n);
    uint32_t* src = data;
    uint32_t* dst = temp.data();

    // Process exactly 32 bits (1 bit per pass)
    for (int bit = 0; bit < 32; ++bit) {
        size_t count0 = 0;
        
        // Count elements with 0 at the current bit
        for (size_t i = 0; i < n; ++i) {
            uint32_t bit_val = (src[i] >> bit) & 1;
            if (bit_val == 0) {
                count0++;
            }
        }
        
        size_t idx0 = 0;
        size_t idx1 = count0; // Elements with bit 1 start after elements with bit 0
        
        // Scatter elements into dst buffer based on bit value
        for (size_t i = 0; i < n; ++i) {
            uint32_t bit_val = (src[i] >> bit) & 1;
            if (bit_val == 0) {
                dst[idx0++] = src[i];
            } else {
                dst[idx1++] = src[i];
            }
        }
        
        // Swap pointers for the next pass
        uint32_t* temp_ptr = src;
        src = dst;
        dst = temp_ptr;
    }
    
    // After 32 passes (an even number), the sorted array will be in the original 'data' pointer, 
    // because we swapped 32 times (src and dst return to original state).
    // Let's verify: 
    // pass 0: src=data, dst=temp -> swap: src=temp, dst=data
    // pass 1: src=temp, dst=data -> swap: src=data, dst=temp
    // ...
    // pass 31: src=temp, dst=data -> swap: src=data, dst=temp
    // Therefore, at the end, src == data. No copy-back is needed.
}
